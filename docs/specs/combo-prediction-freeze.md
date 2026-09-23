# コンボ危険度予測・2024 validation前固定仕様

## 状態

この文書は2020〜2023年だけを使う開発仕様である。2024年validationと2025年holdoutは、
下記の固定条件とGO判定を満たすまで読み込まない。

## 固定する対象

- analysis ID: `combo-development-primary-r1000-v2`
- 対象: 四人麻雀・鳳凰卓・東南戦の東場
- 開発年: 2020、2021、2022、2023
- 予測fold: 2020→2021、2020〜2021→2022、2020〜2022→2023
- 標本: 年ごとに1000牌譜。固定ハッシュ順位、seed `20260923`
- manifest: `outputs/combo-development-sample-2020-2023-random1000.manifest.json`
- manifest schema: v2。候補母集団、選択順位、選択MJAI本文のSHA-256を記録する
- 主ラベル: `ron_eligible`
- 主比較: `conventional_simple - conventional`
- 候補重み: 同一判断内で合計1となる `1 / candidate_count`
- 主指標: out-of-time予測を結合したweighted log loss差
- 副指標: AUC、Brier、ECE、判断内macro concordance、calibration
- 2020〜2023年内の標本数チェックポイント50、100、250、500は診断専用とし、
  結果を見て主標本数1000を変更しない

## 学習条件

目的関数は、候補重み付き平均binary log lossと係数のL2罰則の和とする。

```text
mean_weighted_log_loss + (lambda / 2) * sum(coefficient ** 2)
```

切片は罰しない。solverは解析勾配を渡したL-BFGS-B、最大反復300、最大line search 50、
`gtol=1e-7`、`ftol=0`を初期固定値とする。学習成功はsolverのsuccessだけでは決めず、
有効係数と切片の最大絶対勾配が`1e-7`以下、反復上限未到達、全値有限であることを
必要条件とする。

正則化は次の一回限りの閉じたgridから、baseモデルだけの2021〜2023
expanding-window統合log lossで選ぶ。

```text
{0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
```

差が`1e-6`以内なら大きいlambdaを選ぶ。challengerにも同じlambdaを適用し、
challengerの結果でlambdaを選び直さない。自然下限0を含むため、境界選択になっても
このgridを再拡張しない。

各lambda候補は3つのbase OOT foldすべてで、solver success、反復上限未到達、全値有限、
最大絶対勾配`1e-7`以下を満たす場合だけ選択対象とする。1つでも不合格ならその候補の
統合scoreを欠測として除外し、fold別診断と理由を保存する。全候補が不合格なら解析を
失敗させる。有限scoreとfold診断の適格性はfreeze bundle作成時にも再計算して照合する。

当初のv1 gridは`1e-4`が下端かつ、次点`3e-4`との差がtie許容値を大きく上回った。
2024年を一度も読まない段階の独立監査で下側未確認を指摘されたため、v1出力を監査用に
保存したまま上記v2 gridへ一度だけ拡張した。以後の再探索は禁止する。

条件固定前に、選択lambdaについて最大反復を600に倍増した再fitを行う。全foldで予測の
最大絶対差が`1e-8`以下、log loss差が`1e-10`以下、双方の最大絶対勾配が`1e-7`以下を
要求する。満たさない場合はNO-GOであり、2024年を開かない。

## calibration

予測確率は`1e-15`でclipしてlogitへ変換する。次を区別して報告する。

- calibration-in-the-large: 予測logitの係数を1に固定して推定したoffset。理想値0
- recalibration intercept/slope: `logit(y) = alpha + beta * logit(p)`を重み付き・
  無罰則で同時推定した`alpha`と`beta`。理想値0と1

単一class、定数logit、完全・準完全分離、特異Hessian、line search失敗、未収束を
欠測理由として明示し、有限でない推定値を数値として出さない。clip件数も保存する。

## 巡目bin

判断時の`decision_discard_number`を事前固定した次の4区分に分ける。

- 1〜6巡
- 7〜9巡
- 10〜12巡
- 13巡以降

モデルをbinごとに再学習せず、同じout-of-time予測をmaskして評価する。各binの点推定・
区間に加え、`delta(13+) - delta(1-6)`をlate-minus-early contrastとして報告する。

## 不確実性

- 単位: game cluster
- 反復: 2000
- seed: `20260923`
- モデルはbootstrap内で再fitしない
- 同一replicateのgame multiplicityをoverall、全巡目bin、両モデルで共有する
- 開発年を結合する区間はprediction yearで層化し、各年のgame数を保つ
- late-minus-early区間はreplicateごとの差から直接作り、bin別区間の端点を引かない
- 区間はpointwise percentile 95%。副指標・bin解析は探索的で、多重比較調整なし

## 2024年へ進むGO条件

1. manifest v2の全検査とcache artifactのハッシュが一致する。
2. 人工データで勾配、重みscale不変性、共有bootstrap、年層化、calibration異常系を
   検証したテストが全て通る。
3. 全primary fitが収束条件と倍反復安定性を満たす。
4. 2020〜2023年の主・副結果、巡目bin、calibration、欠測理由を機械可読JSONへ保存する。
5. 実行CLI、Python・NumPy・SciPy版、コード・manifest・cache・出力のSHA-256を保存する。
6. 固定後にコード、特徴、ラベル、lambda選択、指標、標本を変更しない。

GOは結果JSONのstatusだけでは成立しない。結果、development manifest、4年分cache、
`pyproject.toml`、`uv.lock`、全ローカルPython実装、仕様書のSHA-256を含むfreeze bundleを
atomicに作成し、そのbundleを別プロセスで再検証して初めて2024 validationを許可する。
2024専用runnerと開封前pytest証跡コードもbundle作成前に固定し、bundle作成後にコードや
仕様が増減・変更された場合は再びNO-GOとする。

既存cacheはproducer ID `combo-development-primary-r1000-v1`を保持するため、封印前に
現在の抽出・特徴コードで選択4000牌譜を全件再抽出する。4年それぞれについてCSRのshape・
indptr・indices・data、特徴語彙、両ラベル、重み、判断/game IDとindex、巡目bin、source
順位、抽出countがcacheと厳密一致することを要求する。監査の抽出前、抽出後、報告公開
直前の3時点でmanifest、cache、選択原本、全Python・仕様・依存metadataのSHA-256
inventoryとPython・NumPy・SciPy環境が不変であることを確認し、そのPASS報告をfreeze
bundleの必須入力として封印・再検証する。

固定後に変更が必要になった場合、その後の2024再実行は新鮮なvalidationとは扱わず、
2025年を開かない。
