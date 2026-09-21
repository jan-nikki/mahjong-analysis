# 国士無双テンパイ立直抽出仕様

## 目的と対象

`data/archives/v1.2.0-db/YYYY.db`（2009〜2025年）に保存された天鳳XML牌譜から、
立直宣言牌を切った直後の手牌が国士無双テンパイであり、その宣言が
`REACH step="2"` まで成立した事例を全件抽出する。

対象は `logs.num_players = 4`、`logs.is_tonpu = 0` の鳳凰卓半荘である。
`game_type` 列が存在するDBではさらに `game_type = "houou"` を必須とする。
対象行の `log` が欠落している場合は推測せず走査対象外とし、`unavailable_logs`
へ明示的に計上する。「全件」はDBに実際の牌譜本文が保存されている対象行の全件を
意味し、メタデータだけの欠損行数を結果に併記する。

東場だけには限定せず、各半荘の全局を対象にする。`kyoku_index` は牌譜内の
`INIT` を数えた0始まりの通し局番号とする。

## 状態復元

DBの `log` はgzip圧縮またはplain XMLとして読む。各 `INIT` の `hai0`〜`hai3` を
物理牌ID（0〜135）の初期手牌とし、次のイベントを順番どおり反映する。

- `T/U/V/W`: actor 0/1/2/3のツモ牌を加える
- `D/E/F/G`（小文字を含む）: actor 0/1/2/3の打牌を物理牌IDで除く
- `N`: 天鳳のmeld codeを復号し、chi、pon、daiminkan、ankan、kakanで
  手牌から消費した物理牌を除く
- `REACH step="1"`: 宣言者と宣言種別を仮記録する
- 直後の宣言者の打牌: 打牌後の手牌と本人の打牌数を記録する
- `REACH step="2"`: 立直成立を確定し、総立直件数へ加える
- 宣言牌への `AGARI` または `RYUUKYOKU`: 未成立立直として除外する

物理牌が手牌に存在しない、actorが一致しない、`step="2"` に対応する宣言がない、
物理牌IDが重複するなど、復元に推測が必要な牌譜は黙って除外せずエラーにする。
例外として、`AGARI` / `RYUUKYOKU` の同一タグ内で完全に同じ属性名・属性値が
重複する既知のXML不正だけは、2個目以降を除いて再解析する。結果タグの重複属性は
手牌再生に影響せず、値が異なる重複や状態を担う他タグの重複は引き続きエラーにする。
補正した牌譜数は `normalized_xml_logs` に保存する。

`turn` は宣言牌を含む本人の打牌番号（1始まり）とする。

通常の `step=1 → 打牌 → step=2` に加え、実DBに存在する
`打牌 → step=1 → step=2` も扱う。後者は `step=1` の直前が同じactorの打牌で、
その時点の手牌枚数が打牌後枚数に一致する場合だけ受け入れ、現在の13枚を判定する。
この形式の成立数は `post_discard_reach_sequences` に別計上する。

副露して開いた手牌は、その時点で国士・立直とも不可能になるため、以後の物理牌の
増減を追わない。その人の鳴き・打牌イベントの発生は追跡し、他家の手牌復元と
ダブル立直判定は継続する。副露済みactorのREACHはエラーとする。
暗槓だけの門前手牌は引き続き復元し、成立立直の母数に含めるが、国士候補にはしない。
これは、開いた手牌で加槓牌が直前に打牌としても記録される元データ異常があっても、
同局の国士候補を捨てないためでもある。検索対象外の開いた手牌を全面検証する機能ではない。

`is_processed` / `was_error` は変換処理の状態であり、本文の有無と区別する。
対象条件に合い本文が存在すれば、これらのフラグで除外しない。DBはread-only接続する。

## 立直種別

宣言者の最初の打牌であり、かつその `REACH step="1"` より前に牌譜内の `N`
イベントが1件もない場合を `double_riichi` とする。それ以外は `riichi` とする。
このため、他家の副露・暗槓等で最初の巡目が中断された後の初打牌立直を
ダブル立直に数えない。

## 国士無双テンパイ判定

宣言牌を除いた閉じた13枚を既存の `calculate_hand_waits` へ渡し、
`hand_type = "kokushi"` の待ちだけを抽出する。固定面子がある手牌は対象外とする。
中張牌を含む手牌、または么九牌が12種類未満の手牌は必要条件だけで早期除外し、
条件を満たす手牌の最終判定には既存の待ち判定を用いる。

- 么九牌13種が各1枚: `wait_type = "kokushi_13men"`、`waits` は13種
- 么九牌12種とそのうち1種の重複: `wait_type = "kokushi_single"`、
  `waits` は欠けている1種
- 上記以外: 国士無双テンパイではない

## JSON出力

1成立国士立直を `records` の1要素とし、各要素に次を保存する。

- `year`
- `date`
- `log_id`
- `kyoku_index`
- `who`（0〜3）
- `turn`
- `reach_type`（`riichi` / `double_riichi`）
- `wait_type`（`kokushi_single` / `kokushi_13men`）
- `waits`
- `url`（該当プレイヤー・局を指定した天鳳HTML5牌譜URL）

待ち牌は既存コードの牌表記に統一する。`m/p/s` は萬子・筒子・索子、
`E/S/W/N/P/F/C` は東・南・西・北・白・發・中を表す。

`summary` は対象・利用可能・走査牌譜数、総局数、総成立立直件数、
国士立直件数、国士13面待ち件数を持つ。`yearly` に同じ主要件数を年別で保存する。
`records` は year、date、log_id、kyoku_index、who の順で決定論的に並べる。

CLIは完成JSONを `.part` へUTF-8で書いた後に置換し、途中ファイルを完成出力として
残さない。`--max-logs-per-year` は各年をlog ID順に固定した少数検証専用であり、
全件結果とは区別する。

## 実行・検証

プロジェクトルートで実行する。全局を対象とする本機能は、既存の東場限定分析とは
集計母数が異なる。

```powershell
.venv\Scripts\python.exe analysis/find_kokushi_riichi.py --max-logs-per-year 1000 --workers 4 --progress --output outputs/kokushi-riichi-sample.json
.venv\Scripts\python.exe analysis/find_kokushi_riichi.py --workers 12 --progress --output outputs/kokushi-riichi-2009-2025.json
.venv\Scripts\python.exe analysis/validate_kokushi_riichi.py --db-root data/archives/v1.2.0-db --result outputs/kokushi-riichi-2009-2025.json --output outputs/kokushi-riichi-validation.json --workers 12
.venv\Scripts\python.exe -m pytest -q
```

`--progress` は完了した年度の牌譜数・成立立直数・国士件数をstderrへ表示する。
結果の並びは完了順に依存しない。

検算は本番の手牌再生、meld decoder、牌種定数、待ち判定をimportしない。
XMLの状態タグを別経路で読み、Nを一度も行っていない各actorを集合で復元し、
REACH step=2時点の13枚について么九牌の種類数と枚数を直接検査する。
母数の成立立直数はNの有無にかかわらずstep=2を全件数える。
検算結果は年別主要カウンター、抽出キー集合、全出力フィールドを比較し、
検出全件の物理牌ID・13枚の牌種・XML byte offset・入力結果JSONのSHA256を保存する。
結果タグの点数属性は検算に使わないため、その既知の重複正規化も共有しない。

天鳳meld codeの配置は
[tenhou-log-utilsのデコード実装](https://github.com/mthrok/tenhou-log-utils/blob/master/tenhou_log_utils/parser.py)
と照合した。人工テストで各副露種の消費物理牌、通常待ち全156通り、13面待ち、
宣言牌変更、暗槓、複数立直、未成立、表記揺れ、不正な成立列を検証する。
