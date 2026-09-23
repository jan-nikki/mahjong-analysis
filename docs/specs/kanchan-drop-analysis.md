# 嵌張落としに見える河の後の成立リーチ分析

## 目的と対象

`data/processed/riichi-waits-v1` の2009〜2025年、鳳凰卓東南戦・東場・
赤あり四人麻雀の成立リーチ10,706,714件を対象に、本人河の
「嵌張ターツを落としたように見える連続打牌」と宣言直後の待ちを比較する。
主結果は2020〜2025年、全期間は長期確認、各年は年代差確認に用いる。

観測単位と正式な分母は成立リーチ1 recordであり、候補数や
`wait_details`数ではない。通常、ダブル、追っかけを含む既存datasetの
母集団を変更しない。本分析は河の外見上の関連を記述するものであり、
実際の手牌に嵌張ターツがあったこと、打牌意図、因果効果、押し引きの優劣を
示さない。

## 主パターン

同じactorの`actor_discards_before_riichi`にある隣接2打について、次を
すべて満たすものを1候補とする。

- `discard_number`が連続する。
- 両方が数牌で、同じ色である。
- 正規化後の数字差が2である。対象形は13、24、35、46、57、68、79。
- 2枚目の`tsumogiri`が`false`である。
- 1枚目は手出し、ツモ切りのどちらも許容する。

1枚目が手出しならA、ツモ切りならBとする。A/Bは同格であり、どちらも
主パターンへ含める。ツモ切りでも14枚から最弱ブロックを選んだ結果と解釈
できるため、Bを弱い感度群にはしない。2枚目がツモ切りなら非該当とする。

赤5は通常5へ正規化するが、生牌も監査用に保持する。宣言打牌が2枚目の
場合も含む。本人の打牌だけで隣接を判定し、他家イベントや暗槓による
event indexの飛びは問わない。鳴かれた打牌も除去せず、call metadataは
主分類へ使用しない。

## 切り順

各候補に`low_to_high`または`high_to_low`を保存する。さらに数字の5への
距離を`central_distance(rank) = abs(rank - 5)`とし、次へ分類する。

- `inner_first`: 1枚目のほうが5に近い。
- `outer_first`: 2枚目のほうが5に近い。
- `symmetric`: 距離が同じ。46形だけが該当する。

具体的な内側先行／外側先行は次のとおり。

| 形 | inner first | outer first |
|---|---|---|
| 13 | 3→1 | 1→3 |
| 24 | 4→2 | 2→4 |
| 35 | 5→3 | 3→5 |
| 46 | 4→6、6→4ともsymmetric | - |
| 57 | 5→7 | 7→5 |
| 68 | 6→8 | 8→6 |
| 79 | 7→9 | 9→7 |

主仮説は`inner_first`の好形率、筋待ち率、両面率が`outer_first`より高い
ことである。46形はinner/outer比較から除き、方向別に集計する。

## record分類、代表候補、距離

全候補を見てrecordを`none / a_only / b_only / both`の排他的4区分へ
1回だけ計上する。パターンありは`a_only + b_only + both`である。

切り順の正式比較で1 recordを複数回数えないため、2枚目の
`discard_number`が最大の候補を代表候補とする。代表候補の形、方向、
inner/outer/symmetric、A/Bを使って排他的に集計する。全候補における
切り順は監査情報に残すが、主要率の分母には使わない。

```text
distance = riichi_discard_number - representative second discard_number
```

宣言打牌が代表候補の2枚目ならdistanceは0、パターンなしはnullである。

## 待ち指標

`src/mahjong_analysis/riichi_wait_quality.py`の`evaluate_wait_quality()`を
唯一の正本として再利用する。

- `contains_suji`: standard/ryanmenまたは同色のstandard/tanki解釈2種が
  3または6離れるノベタン系を1つ以上含む。
- `good_wait`: 宣言後の純手牌とfixed meldを正規化して自己所有分を除き、
  重複しないformal wait tileの`4 - self_owned`合計が5枚以上。
- `contains_ryanmen`: 複合待ちを含め、両面解釈が1つ以上ある。
- `self_excluded_wait_copies`: 自己所有を除いた形式待ち枚数。

筋待ち率は嵌張候補の中間牌との関係ではなく、リーチ待ち全体に筋待ち解釈が
存在するrecordの率である。好形は他家、河、ドラ表示、実山残枚数を控除しない。

## 比較群と分母

全期間、2020〜2025小計、年度、exact宣言打牌番号について次を保存する。

- 全record
- パターンあり／なし
- Aのみ／Bのみ／両方
- 代表候補のinner first／outer first／symmetric
- 代表候補のlow-to-high／high-to-low
- 代表候補の形×方向
- 代表候補のA/B×inner/outer/symmetric

各record-level率の分母はその群のrecord数とし、分母0は`null`とする。
主要差はパターンあり−なし、およびinner first−outer firstのpercentage points。
各形では両方向を保存し、13・24・35・57・68・79はinner−outer、46は
high-to-low−low-to-highを保存する。特に24形の4→2と2→4を直接比較する。

対子落とし型との重複は主群から除外しない。既存`detect_toitsu_drop()`で
対子落とし型あり／なしへ分割した補助結果も保存し、対子落とし型なし層でも
同じ比較を確認できるようにする。

## 巡目・距離・年度

巡目の正本は卓全体の巡目ではなく本人の`riichi_discard_number`である。
各exact宣言打牌番号で同じ比較を保存する。1打目はパターン群分母0となる。
年度×宣言打牌番号も年度ブロック内に保存する。代表候補から宣言までの
exact distance別にも同じ分類を保持する。率の平均ではなく整数countの合算から
全体率を再計算する。

## ストリーミングと検証

manifest記載の17年度だけを年度ごとにstreamingし、annual fileのsize/SHA256、
DTO、年度行数、totalを検証する。Windows互換のprocess workerを利用できる。
結果は年度、巡目、距離を数値昇順へmergeし、worker完了順に依存させない。

全record集計は`riichi-wait-quality-v1.json`のoverall、年度、exact宣言打牌番号の
品質値と完全一致させる。入力・baseline・コードが実行中に変化した場合は
出力しない。既存baseline、canonical dataset、対子落とし結果は変更しない。

## 監査

stable key `(relative_source_path, start_kyoku_line, reach_line)`で決定論的に
最大30 recordを選ぶ。none、A、B、both、inner、outer、symmetric、両方向、
宣言打牌完了、赤5、複数候補、成立前に鳴かれた打牌を固定カテゴリとする。
本番後に元MJAIをread-onlyで再読込し、河、候補、代表候補、距離、切り順を
独立に再計算する。

取りこぼし対策として、人工テストでは全34牌種ペア×1枚目・2枚目の
手出し／ツモ切りを独立oracleと照合する。

## 出力

正式結果を`research/results/kanchan-drop-analysis-v1.json`へ保存する。
Work向けには次を作る。

```text
outputs/work/kanchan-drop-analysis/
  handoff.md
  article-brief.md
  result-summary.json
  work-prompt.txt
  figures/
```

Workの見出しは2020〜2025年を主とし、全期間を長期確認とする。

## 必須不変条件

- 4分類の合計が全record数。
- パターンあり＋なしが全record数。
- 候補数分布の合計が全record数。
- 代表order、direction、shape×direction、A/B×orderの各合計がパターン件数。
- 距離別合計がパターン件数。
- 対子落とし型あり／なしの合計が全record数。
- 年度合計、宣言打牌番号合計がoverall。
- formal待ち指標の分母が各群record数。
- serial/parallelの内容と順序が一致。
- malformed JSON、年度不一致、行数不一致、worker errorを黙って飛ばさない。
- 全record品質が既存baselineとoverall・年度・宣言打牌番号で一致する。
