import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

INPUT_PATH = Path(r"outputs\ryuuiisou-capture-targets.json")
OUT_DIR = Path(r"outputs\ryuuiisou-final-135")
LOG_PATH = OUT_DIR / "capture-log.json"
FAIL_PATH = OUT_DIR / "failed.json"

VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080

CLICK_X = 960
CLICK_Y = 630

# 概算位置より20ステップ手前から開始
START_BACK = 20

# 最大45ステップ進めて和了結果画面を探す
MAX_STEPS = 45

STEP_WAIT_MS = 450
START_WAIT_MS = 1800
MAX_LOAD_RETRIES = 4

# 最初の10件だけテスト
all_data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
data = all_data

OUT_DIR.mkdir(parents=True, exist_ok=True)


def classify(png_bytes):
    """
    start : 天鳳の黒い「牌譜を再生」画面
    popup : 和了結果の黒いポップアップ
    table : 通常の卓画面
    """

    image = Image.open(BytesIO(png_bytes)).convert("L")

    image = image.resize((240, 135))

    pixels = list(image.getdata())

    nonblack_ratio = sum(1 for p in pixels if p > 10) / len(pixels)

    # 1920x1080画面の中央付近
    center = image.crop((26, 36, 133, 78))

    center_pixels = list(center.getdata())

    center_dark_ratio = sum(1 for p in center_pixels if p < 40) / len(center_pixels)

    if nonblack_ratio < 0.15:
        return (
            "start",
            nonblack_ratio,
            center_dark_ratio,
        )

    if center_dark_ratio > 0.60:
        return (
            "popup",
            nonblack_ratio,
            center_dark_ratio,
        )

    return (
        "table",
        nonblack_ratio,
        center_dark_ratio,
    )


def open_viewer(page, url):
    """
    URLを開き、「牌譜を再生」を押して
    卓画面になるまで試す。
    """

    for attempt in range(1, MAX_LOAD_RETRIES + 1):
        print(f"    load attempt {attempt}/{MAX_LOAD_RETRIES}")

        page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=30000,
        )

        page.wait_for_timeout(1200)

        page.mouse.click(CLICK_X, CLICK_Y)

        page.wait_for_timeout(START_WAIT_MS)

        png = page.screenshot(full_page=False)

        state, nb, dark = classify(png)

        print(f"    state={state} nonblack={nb:.3f} center_dark={dark:.3f}")

        if state != "start":
            return png

        # まだ起動画面なら再クリック
        page.mouse.click(CLICK_X, CLICK_Y)

        page.wait_for_timeout(START_WAIT_MS)

        png = page.screenshot(full_page=False)

        state, nb, dark = classify(png)

        if state != "start":
            return png

    return None


results = []
failed = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    page = browser.new_page(
        viewport={
            "width": VIEWPORT_WIDTH,
            "height": VIEWPORT_HEIGHT,
        },
        device_scale_factor=1,
    )

    total = len(data)

    for index, target in enumerate(data, 1):
        approximate_tj = target["tj_base"]

        start_tj = max(
            0,
            approximate_tj - START_BACK,
        )

        url = (
            f"https://tenhou.net/5/"
            f"?log={target['log_id']}"
            f"&ts={target['kyoku_index']}"
            f"&tw={target['who']}"
            f"&tj={start_tj}"
        )

        filename = (
            f"{index:03d}"
            f"_{target['year']}"
            f"_{target['log_id']}"
            f"_k{target['kyoku_index']:02d}"
            f".png"
        )

        out_path = OUT_DIR / filename

        print()
        print("=" * 70)

        print(f"[{index}/{total}] {target['year']} {target['log_id']}")

        print(f"approx_tj={approximate_tj} start_tj={start_tj}")

        first_png = open_viewer(page, url)

        if first_png is None:
            failed.append(
                {
                    "index": index,
                    "log_id": target["log_id"],
                    "reason": "viewer_start_failed",
                }
            )

            print("  FAILED: viewer start")

            continue

        previous_table_png = None
        success = False
        current_png = first_png

        for step in range(MAX_STEPS + 1):
            state, nb, dark = classify(current_png)

            print(
                f"  step={step:02d}"
                f" state={state}"
                f" nonblack={nb:.3f}"
                f" center_dark={dark:.3f}"
            )

            if state == "table":
                # 直近の通常卓画面を保持
                previous_table_png = current_png

            elif state == "popup":
                if previous_table_png is None:
                    print("  popup appeared before table frame")
                    break

                # ポップアップ直前の卓を保存
                out_path.write_bytes(previous_table_png)

                print("  FOUND POPUP")

                print("  SAVED PREVIOUS:", out_path.name)

                results.append(
                    {
                        "index": index,
                        "year": target["year"],
                        "log_id": target["log_id"],
                        "kyoku_index": target["kyoku_index"],
                        "who": target["who"],
                        "start_tj": start_tj,
                        "steps": step,
                        "file": filename,
                    }
                )

                success = True
                break

            # 天鳳ビューアで1ステップ進む
            page.keyboard.press("6")

            page.wait_for_timeout(STEP_WAIT_MS)

            current_png = page.screenshot(full_page=False)

        if not success:
            failed.append(
                {
                    "index": index,
                    "year": target["year"],
                    "log_id": target["log_id"],
                    "kyoku_index": target["kyoku_index"],
                    "start_tj": start_tj,
                    "reason": "result_popup_not_found",
                }
            )

            print("  FAILED: popup not found")

    browser.close()


LOG_PATH.write_text(
    json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

FAIL_PATH.write_text(
    json.dumps(
        failed,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 70)
print("TEST FINISHED")
print("success:", len(results))
print("failed :", len(failed))
print("output :", OUT_DIR.resolve())
