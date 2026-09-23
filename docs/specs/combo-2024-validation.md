# コンボ危険度予測・2024年独立validation仕様

> **Status: superseded（履歴資料）**
>
> この文書は2024年独立validationの実行条件を固定した履歴であり、現在の作業計画ではない。
> 最終判断は
> [`combo-nonutility-assessment.md`](../validation/combo-nonutility-assessment.md) を正本とし、
> 単純コンボ戦術の追加調査は終了している。

## 前提

2024年を選択・抽出する前に、`combo-development-primary-r1000-v2`のfreeze bundleが
GOであり、bundleのpayloadおよび全記録ファイルのSHA-256再検証に成功しなければ
ならない。さらに、固定後の全pytestを無選択で実行し、exit code 0、collection件数、
結果件数、transcript SHA-256、`tests/**/*.py`全ファイルのSHA-256をbundleへ紐づけた
preflight reportとして保存・再検証する。2024年を一度でも評価した後は、特徴、ラベル、標本、lambda、
学習器、停止基準、主指標、成功判定を変更しない。変更が必要ならその2024年結果は
独立validationではなくなり、2025年を開かない。

実行順は、bundle検証、preflight検証、2020〜2023年cache検証、固定モデルfitと収束
gate、2024開封markerの永続化、2024標本列挙の順に固定する。開封markerは2024の
ディレクトリを列挙する前にatomicかつ上書き不可で保存する。これより前に失敗した場合は
2024未開封、これより後に失敗した場合は結果JSONがなくても2024開封済みとして扱う。

## 2024年標本

- 対象: 四人麻雀・鳳凰卓・東南戦の東場
- 牌譜数: 1000
- 選択: 開発標本と同じ
  `lowest sha256(namespace, seed, relative_source), v1`
- seed: `20260923`
- manifest: schema v2
- manifestには候補母集団、選択順位、選択MJAI本文のSHA-256を保存する
- 選択MJAI本文は抽出時の各ファイル読込前・直後、および全件抽出後かつcache公開前に
  manifestのSHA-256と再照合し、変化があれば停止する
- 2024年manifestを作る処理は2020〜2023年のfreeze検証に成功した後だけ実行する
- manifest、cache、結果、receiptは完成した一時ファイルをatomicかつ上書き不可で公開する
- 2025年は選択、列挙、抽出、目視確認をしない

## 学習と予測

- 学習データ: 固定済み2020〜2023年cacheの全1000牌譜/年
- validationデータ: 2024年1000牌譜のみ
- ラベル: `ron_eligible`
- 候補重み: `1 / candidate_count`
- base: `conventional`
- challenger: `conventional_simple`
- 特徴語彙: 学習データだけから作り、2024年固有特徴は未知特徴として係数0にする
- lambda: v2 freeze bundleの`selected_l2`を両モデルへ同一に使用する
- 2024年でlambda選択、特徴選択、閾値選択、再calibrationを行わない
- 目的関数: weighted mean log loss + `lambda / 2 * ||beta||^2`
- solver: 解析勾配付きSciPy L-BFGS-B
- 最大反復300、`gtol=1e-7`、`ftol=0`、line search最大50、history 20
- 切片は無罰則
- solver success、反復上限未到達、全値有限、最大絶対勾配`1e-7`以下を全て要求する
- 不合格時は解析失敗として止め、条件を緩めて同じ2024年を再評価しない

## 主評価

estimandは2024年の固定予測に対する
`conventional_simple - conventional`のweighted log loss差とする。game単位の対応
bootstrapを2000回、seed `20260923`で行い、モデルはbootstrap内で再fitしない。

次の両方を満たした場合だけ主validation成功とする。

1. log loss差の点推定が0未満
2. pointwise percentile 95%区間の上端が0未満

これは固定済み学習手続きから得た予測に条件づけた区間であり、開発時のlambda選択を
bootstrap内で再現する区間ではない。

## 副評価と診断

- AUC、Brier、ECE、判断内macro concordance
- calibration-in-the-large
- 同時推定recalibration intercept/slope
- 固定巡目bin: 1〜6、7〜9、10〜12、13巡以降
- `delta(13+) - delta(1-6)`の対応contrast
- clip件数、calibration異常状態、game・判断・候補行数

calibrationは診断値としてのみ報告し、2024年の予測確率を補正しない。副評価と巡目binは
探索的であり、主validationの成否を上書きしない。各指標では同じgame multiplicityを
両モデル・全binで共有する。

## 出力と封印

- 2020〜2023年freeze bundleのSHA-256
- preflight report、pytest transcript、test inventoryのSHA-256
- 2024 manifest・cache・結果JSONのSHA-256
- 実行CLI、Python・NumPy・SciPy版
- 抽出、特徴、ラベル、学習、評価に到達する全ローカルPythonファイルのSHA-256
- 全テスト結果
- `validation_2024_touched=true`、`holdout_2025_touched=false`
- 結果JSON自身のSHA-256は、結果とは別の上書き不可なreceiptに保存する

結果JSONのatomic公開後、receipt公開前に停止した場合に限り、`--recover-receipt`で
receiptだけを復旧してよい。復旧時はfreeze bundle、preflight report、開封marker、
結果の全必須nested schema、固定optimizer診断、語彙、件数、全metrics・calibration・
巡目bin・bootstrapの値域と内部整合、および結果が参照するmanifest・cache等のpathと
SHA-256を再検証し、rawデータの列挙・抽出、モデルfit、予測、bootstrapを一切再実行しない。結果JSONと
receiptがともに存在する場合は検証のみを行い、通常実行は既存の結果JSONを拒否する。

結果が成功でも不成功でも、そのまま報告する。2025年を開く判断は別途明示的に行い、
少なくとも2024年結果を見た再調整がないことを確認する。
