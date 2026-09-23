# 親子別・成立リーチ平均打点の要因分解仕様（第2回）

## 目的

第1回の親子別平均打点集計で観測した「親の和了時平均打点 − 子の和了時平均打点」を、
次の3要因へ記述的に分解する。

- `R`（role / payment rule）: 同じ和了方法・同じ基本点tierに親または子の支払い規則を適用する差
- `M`（method mix）: ツモ／ロン構成比の差
- `B`（basic-point mix）: ツモ／ロン別に見た基本点tier構成の差

3要因の変更順序に依存しないよう、子を基準、親を比較対象とする3要因Shapley分解を使う。
これは親子差の会計的・記述的な分解であり、「親であることが手を高くする」といった
因果効果の推定ではない。

## 入力と対象範囲

入力は第1回production runの次の2成果物とし、MJAI原本は再走査しない。

```text
outputs/dealer-child-riichi-points/summary-v1.json
outputs/dealer-child-riichi-points/records-v1.jsonl.gz
```

要求する入力identityは次のとおりである。

- `analysis_name == "dealer-child-riichi-points-v1"`
- summaryの`schema_version == 2`
- repositoryが`NikkeTryHard/tenhou-to-mjai`
- release tagが`v2.0.0`
- formatが`MJAI JSON Lines`
- scopeが`rule_code == "00a9"`, `aka_flag is true`, `bakaze == "E"`,
  `riichi_population == "established"`
- `primary_years`が2020〜2025年の重複しない昇順6年
- 対象年が2009〜2025年の重複しない昇順17年

第1回と同じく、四人麻雀の鳳凰卓東南戦、rule code `00a9`、赤牌あり、東場を対象とする。
観測単位は成立リーチ1件だが、本分解の推定対象は、そのうち`outcome == "win"`である
和了リーチの`hand_points`条件付き平均とする。`other_win`と`draw`は入力母集団の整合性検証には
使用するが、打点分解の分母には入れない。

主期間は2020〜2025年、長期確認は2009〜2025年とする。年度別結果も出力し、年代によって
分解結果が変わらないか確認する。主分析は通常リーチとダブルリーチをともに含める。

1件の和了リーチについて使用する入力フィールドは最低限次である。

- identity: `year`, `source_path`, `kyoku_index`, `reach_event_index`, `actor`
- 親子: `oya`, `actor`
- リーチ種別: `reach_type`
- 結果: `outcome`, `win_method`
- 得点: `hand_points`
- 原本追跡: `start_kyoku_line`, `reach_line`, `reach_accepted_line`, `hora_line`

親は`actor == oya`、子は`actor != oya`とする。`hand_points`は第1回と同じく、本場と供託を
除いた和了手本体について、和了者以外が支払った合計点である。ツモでは3人の支払い合計を
用い、`settlement_gain`は使用しない。

## 基本点tierと標準得点変換

### tierの定義

各和了を、符・翻・役名ではなく、点数計算上の基本点`b`で表す。同じ`b`になる異なる
符翻構成は同一tierとする。非満貫tierの候補は、標準的な符

```text
20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110
```

と1〜4翻から得る`fu * 2 ** (han + 2) < 2000`の重複しない値とする。限界手は次を使う。

| tier | 基本点 `b` |
|---|---:|
| 満貫 | 2,000 |
| 跳満 | 3,000 |
| 倍満 | 4,000 |
| 三倍満 | 6,000 |
| 数え役満・役満 | 8,000 |
| 複数役満 | `8,000 * k`（`k >= 2`の整数） |

この候補生成は、観測された支払い額を標準得点表へ対応づけるためだけに行う。符、翻、役、
ドラを復元せず、20符・25符を含む個々の符翻組合せがその牌姿で成立するかも再判定しない。
30符4翻および60符3翻は基本点1,920として扱い、切り上げ満貫にはしない。

### 支払い式

`ceil100(x) = 100 * ceil(x / 100)`とする。本場・供託・リーチ棒を含めず、標準得点変換
`P(role, method, b)`を次で定義する。

```text
P(親, ロン, b)   = ceil100(6b)
P(親, ツモ, b)   = 3 * ceil100(2b)
P(子, ロン, b)   = ceil100(4b)
P(子, ツモ, b)   = ceil100(2b) + 2 * ceil100(b)
```

ツモは各支払いを100点単位へ個別に切り上げてから合計する。したがって、非満貫では
親子比が常に厳密な1.5になるとは限らない。

### 観測値からの逆変換

和了recordごとに、観測された`(role, win_method, hand_points)`について

```text
P(role, win_method, b) == hand_points
```

を満たす基本点候補を列挙する。各候補を親・子双方へ順方向変換し、得られる
`(親換算点, 子換算点)`が候補間で同一であることを要求する。
複数役満候補の`k`は観測点から導ける有限の上限まで列挙する。

- 候補0件: 非標準得点または入力不整合としてrun全体を失敗させる
- 候補が複数でも親・子双方の換算結果が同一: 同じ換算tierとして受理する
- 候補間で親または子の換算点が異なる: 曖昧な逆変換としてrun全体を失敗させる
- 候補1件: その`b`を当該和了の固定tierとする

曖昧なrecordを除外したり、最小／最大候補を暗黙に選んだりしない。エラーには少なくとも
`source_path`, `kyoku_index`, `actor`, role, `win_method`, `hand_points`, 候補一覧を含め、
MJAI原本へ戻れるようにする。同じ基本点に畳み込まれる複数の符翻構成は曖昧とはみなさない。

### 親子反実仮想

各和了の反実仮想点は、推定した基本点`b`と観測された`win_method`を固定し、`role`だけを
親または子へ差し替えて`P`を再計算した値とする。

例えば満貫tierでは、子ロン8,000点を親ロン12,000点へ、子ツモ8,000点を親ツモ
12,000点へ変換する。これは同一牌姿の役・符を再計算する処理ではなく、得点tierと
和了方法を固定した支払い規則の置換である。他家の親子構成、押し引き、和了確率、
ツモ／ロンの発生自体は変更しない。

## 8個の標準化平均

期間および分析対象（全リーチ／通常リーチのみ）ごとに、親を`D`、子を`C`とする。
role `s`の和了件数を`N_s`、方法`m`の件数を`N_s,m`、方法`m`かつ基本点tier`b`の件数を
`N_s,m,b`とし、次を定義する。

```text
pi_s(m)      = N_s,m / N_s
q_s(b | m)   = N_s,m,b / N_s,m
```

方法構成の供給元を`u`、方法内tier構成の供給元を`v`として、8個の平均を

```text
mu[r,u,v] = sum_m pi_u(m) * sum_b q_v(b | m) * P(r, m, b)
```

で求める。`r`, `u`, `v`はそれぞれ`C`または`D`である。B要因を方法内の条件付き分布と
定義することで、方法とtierの実データ上の関連を保ち、両端は観測平均へ厳密に一致する。

```text
mu[C,C,C] = 子の観測平均hand_points
mu[D,D,D] = 親の観測平均hand_points
```

全8セルを計算するには、親・子それぞれでツモとロンの両stratumに1件以上の和了が必要である。
必要な分母が0ならその期間を部分的に出力せず、入力または指定期間が分解不能として失敗させる。

実装では件数と整数点から`fractions.Fraction`等で8平均とShapley値を厳密に計算し、
恒等式を丸め前に検証する。JSONには丸め前の浮動小数値を出し、Markdown表示の丸め値を
後続計算へ再利用しない。

### 100点単位切り上げの効果

同じ方法・同じ基本点tierを親規則と子規則で評価した点を、和了record `i`について
`D_i`, `C_i`とする。100点単位の個別切り上げによって厳密な1.5倍から生じる差を

```text
e_i = D_i - (3 / 2) * C_i
```

と定義する。観測親record由来、観測子record由来、および両者を合わせたpooledの3種類について、
それぞれ`mean(e_i)`を出力する。これは親子の観測平均差そのものではなく、同一の方法・tierで
支払い規則だけを交換したときの丸め差である。計算とJSONの分子・分母は`Fraction`で厳密に保持し、
表示用小数から再計算しない。

## 3要因Shapley分解

ビット`i,j,k`を順に`R,M,B`とし、0は子由来、1は親由来とする。

```text
v000 = mu[C,C,C]    v100 = mu[D,C,C]
v010 = mu[C,D,C]    v110 = mu[D,D,C]
v001 = mu[C,C,D]    v101 = mu[D,C,D]
v011 = mu[C,D,D]    v111 = mu[D,D,D]
```

子から親へ3要因を変更する全6順序の限界寄与を平均し、各寄与を次で求める。

```text
phi_R = 1/3 * (v100 - v000)
      + 1/6 * ((v110 - v010) + (v101 - v001))
      + 1/3 * (v111 - v011)

phi_M = 1/3 * (v010 - v000)
      + 1/6 * ((v110 - v100) + (v011 - v001))
      + 1/3 * (v111 - v101)

phi_B = 1/3 * (v001 - v000)
      + 1/6 * ((v101 - v100) + (v011 - v010))
      + 1/3 * (v111 - v110)
```

次の効率性恒等式を厳密に満たさなければならない。

```text
phi_R + phi_M + phi_B == v111 - v000
```

各寄与は負になり得る。差に対する構成比`phi_x / (v111 - v000)`も出力するが、0〜1へ
丸め込まない。観測差が0なら構成比はJSON `null`、Markdown `N/A`とする。

## 通常リーチのみの感度分析

主分析と同じ処理を`reach_type == "riichi"`だけに限定して再実行する。ダブルリーチを
単に最終表から引くのではなく、親子×ツモ／ロン×tierの件数、8平均、Shapley寄与をすべて
通常リーチsubsetから再計算する。

出力では主分析と感度分析について、`phi_R`, `phi_M`, `phi_B`および親子観測差を並記し、
感度分析からダブルリーチrecordが0件であることを検証する。感度分析も必要な4つの
親子×方法stratumのいずれかが空ならfail closedとする。

## 実装APIとCLI

共通ロジックは`src/mahjong_analysis/dealer_child_score_decomposition.py`、実行コードは
`analysis/analyze_dealer_child_score_decomposition.py`へ置く。

公開APIは少なくとも次の責務を分離する。

- `score_points(role, win_method, basic_points) -> int`: 標準支払い式の順方向変換
- `infer_basic_points(role, win_method, hand_points) -> int`: 一意なtierの逆変換
- record loader: gzip JSON Linesの型・順序・identity・件数検証
- period aggregator: 親子×方法×tierの十分統計を1回の走査で作る
- standardized means: 十分統計から8平均を作る
- Shapley decomposition: 8平均から3寄与を作る
- serializer / renderer: 同一のin-memory resultからJSONとMarkdownを作る

既定CLIは次の形とする。

```text
uv run python analysis/analyze_dealer_child_score_decomposition.py --all
```

CLIは`--all`または`--years YEAR [YEAR ...]`のいずれかを必須とし、入力summary、入力records、
出力JSON、出力Markdownをpath optionで上書き可能にする。選択年は重複を拒否して昇順へ
正規化する。2009〜2025年をすべて選択した場合に主期間と長期期間を出し、それ以外でも
選択期間と年度別結果は出す。

recordsはtupleやlistへ実体化せず、選択期間、各年、主期間、長期期間の各々について、
全リーチと通常リーチのみの固定個数の十分統計へ1recordずつ同時加算する。入力件数に比例する
全件identity setは持たず、直前のcanonical keyと現在局の最大4 actorだけを保持する。このため
集計器のメモリ使用量は入力record数に依存しない。入力summaryとgzip recordsのSHA-256も
ファイル全体をメモリへ読み込まず、byte chunkを逐次hashへ投入して計算する。

## 出力

既定出力は次とする。

```text
outputs/dealer-child-score-decomposition/summary-v1.json
outputs/dealer-child-score-decomposition/summary-v1.md
```

JSONには最低限次を含める。

- analysis name、schema version、入力dataset identity、選択年
- 入力summaryとrecordsの論理pathおよびSHA-256
- 基本点候補、得点変換式、反実仮想の定義
- 読み込んだ全record数、選択年record数、選択年和了record数、変換成功件数、親子×方法別件数
- 観測親子×ツモ／ロン別の件数と基本点tier分布。複数基本点候補が同じ親子換算になる場合は、
  候補基本点一覧、親換算点、子換算点を持つ1つのcanonical conversion tierとして記録する
- 子と親の観測平均および観測差
- `v000`〜`v111`の8標準化平均と、各セルのrole／method source／tier source
- `R`, `M`, `B`のShapley寄与、観測差に対する構成比、効率性残差
- 観測親由来、観測子由来、pooledの`mean(D_i - 1.5 * C_i)`丸め効果
- 通常リーチのみの同一結果
- 主期間、長期期間、選択期間、年度別結果
- 逆変換成功件数と曖昧／非対応件数（成功runでは後二者は0）

Markdownは記事確認用として、選択期間、主期間、長期期間の各々について、全リーチと通常リーチ
のみを分け、親子×方法件数、観測平均、8平均、3寄与、寄与構成比、効率性残差、3種類の丸め効果、
canonical conversion tier分布を表で示す。
JSONとMarkdownは同じin-memory resultから生成し、timestamp、処理時間、絶対pathを含めない。
JSONは`sort_keys=True`, UTF-8, `allow_nan=False`で決定的に出力する。2成果物は第1回runnerと
同様に一時fileへ完成させて検証後に1 bundleとして公開する。2個目の置換を含む途中失敗では、
既存の両成果物をrun前のbytesへ戻し、片方だけ新しくなった状態を残さない。`OSError`だけでなく
`KeyboardInterrupt`等の`BaseException`でもrollbackを試みる。rollback自体に失敗した場合は
復旧用backupを絶対に削除せず、そのpathをエラーへ含める。

## 入力検証

recordsは先頭から末尾まで読み、少なくとも次を検証する。

- gzipの全memberを読み切れ、各行がJSON objectである
- required fieldが存在し、boolを整数として受理せず、列挙値とnull条件が第1回schemaに合う
- canonical key
  `(year, source_path, kyoku_index, reach_event_index, actor)`がstream全体で厳密な昇順である。
  直前key以下なら、重複か逆順かを問わずrun全体を失敗させる
- `source_path`の年度と`year`が一致し、選択年・record順が決定的である
- `actor`と`oya`が0〜3で、そこから再計算した親子区分を使う
- `outcome == "win"`なら`win_method`と正の100点単位の`hand_points`が非nullである
- 非和了なら和了専用fieldがnullである
- 選択外を含む全17年について、成立リーチ総数、結果別件数、親子別件数、和了件数、ツモ／ロン件数、
  `hand_points`合計、`settlement_gain`合計がsummaryの年度結果と一致する
- 通常リーチsubsetについても、同じ整数集計がsummaryの
  `sensitivity_normal_riichi_only`と一致する
- Article 1の`selected`, `primary_2020_2025`, `long_term_2009_2025`について、periodのyears、
  integer counts/sums、および保存済みの方法別・全体meanが全年度のrecord再集計と一致する
- Article 2の選択期間は、選択した年度の十分統計の合算と一致する

summaryに保存された表示済み平均は計算入力にせず、integer countとrecordの
`hand_points`から再計算して照合だけに使う。1件でも不一致ならskipせずrun全体を失敗させる。

## 検証計画

人工recordで最低限次をテストする。

- 親ロン、親ツモ、子ロン、子ツモの順方向得点変換
- 非満貫で各支払いを個別に100点単位へ切り上げること
- 30符4翻／60符3翻相当を切り上げ満貫にしないこと
- 満貫、跳満、倍満、三倍満、役満、複数役満tier
- 一意な逆変換と順方向へのround trip
- 候補0件と複数候補のfail-closed、および原本位置を含むエラー
- roleだけを交換し、methodと基本点tierを保持すること
- 8平均の`v000`と`v111`が観測平均へ一致すること
- 手計算可能な小標本で`phi_R`, `phi_M`, `phi_B`が一致すること
- 3寄与の合計が観測親子差へ厳密に一致すること
- 100点単位切り上げ差を親由来、子由来、pooledで厳密分数として再現すること
- 負および100%超の寄与構成比を丸め込まず、観測差0では構成比を`null`とすること
- ダブルリーチを含む主分析と除く感度分析
- 空の親子×方法stratumを拒否すること
- malformed field、重複record、summaryとの件数・合計不一致を拒否すること
- concatenated gzip memberを全件読めること
- single-pass iterableを再走査せず、入力順を逆転しても複数年度の十分統計と結果が一致すること
- 年度結合結果と全期間直接集計が一致すること
- 入力の論理path、SHA-256、全件・選択件・和了・変換件数がJSONへ一致すること
- JSONとMarkdownが同じ結果を表し、再実行でbytesが一致すること
- 2成果物の途中公開失敗時に両方がrun前の状態へ戻ること
- 公開の各置換直後に`KeyboardInterrupt`が発生しても両成果物を復旧し、rollback失敗時は
  旧artifactのbackup bytesとpathを保持すること

production run後は、親子×ツモ／ロン×非満貫／限界手から実牌譜を抽出し、第1回recordの
原本位置を用いて`hora.deltas`を目視確認する。また、全和了recordについて
`P(observed role, observed method, inferred b) == hand_points`が成立すること、親子の観測平均と
通常リーチのみ平均が第1回summaryへ一致することを機械的に確認する。

## 今回扱わないもの

- 役、符、翻、ドラ、赤ドラ、裏ドラ、一発の復元または要因分解
- 同じ牌姿を親子として再採点する完全な和了形再生
- 親であることによる手作り、押し引き、他家行動、和了率、ツモ率への因果効果
- 成立リーチ1件当たり期待得点（本分析は和了時平均打点が対象）
- 本場、供託、リーチ棒、順位価値を含む収支
- プレイヤー強度、局面、巡目、待ち、先制／追っかけ等での調整
- Shapley寄与の信頼区間、bootstrap、仮説検定
- 3要因以外の交互作用を独立した第4要因として解釈すること

Shapley値は8平均に含まれる交互作用を3要因へ規則的に配分するが、各寄与を構造的・因果的な
効果とは呼ばない。記事では「親子の観測平均差のうち、この標準化規則で各要因へ割り当てた
点数」と表現する。
