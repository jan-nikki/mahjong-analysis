# 成立リーチ待ちbaseline集計仕様

## 目的

canonical dataset `data/processed/riichi-waits-v1` を用いて、全成立リーチの
待ち牌数、待ち形解釈、両面性、リーチ宣言打牌番号の記述統計を確定する。
この結果は、後続の対子落とし・カンチャン落としに見える河、および筋の危険度を
比較する際のbaselineとする。

本分析では河パターンの抽出、放銃危険度、他家手牌・山残枚数・フリテンを考慮した
live wait、年度差の統計的検定は扱わない。「良形」という新しい分類も定義しない。

## 入力datasetと対象範囲

入力の正本は次の完成datasetである。

| 項目 | 値 |
|---|---|
| root | `data/processed/riichi-waits-v1` |
| manifest | `data/processed/riichi-waits-v1/manifest.json` |
| `dataset_name` | `riichi-waits-v1` |
| `schema_version` | 1 |
| mode | `full` |
| years | 2009〜2025 |
| source | `NikkeTryHard/tenhou-to-mjai`, release `v2.0.0` |
| scope | `rule_code=00a9`, `aka_flag=true`, `bakaze=E` |
| annual files | `2009.jsonl.gz`〜`2025.jsonl.gz` |
| manifest `scanned_files` | 2,500,236 |
| manifest `target_games` | 2,500,235 |
| manifest `east_kyokus` | 14,434,809 |
| manifest `output_records` | 10,706,714 |

本分析の主期間は2009〜2025とし、2020〜2025だけを既定期間にはしない。

集計開始前にcompleted manifestのschema、dataset identity、mode、scope、対象17年度を
検証する。年度artifactの存在、compressed size、SHA256もmanifest値と照合する。
annual JSONLを全件decodeするdeep validationは集計前に別passでは実行せず、後述の
1-pass streaming内でrecord validation、年度行数検証、aggregationを同時に行う。
年度ごとの読込件数はmanifestの`years[].output_records`と一致しなければならない。
全年度の読込件数は`manifest.totals.output_records`および
`manifest.totals.established_riichis`と一致し、canonical runでは10,706,714件でなければ
ならない。不一致、欠損、壊れたJSON、不正recordはskipせずrun全体を失敗させる。

## 使用する既存field

schema v1に実在する次のfieldだけを入力として用いる。集計実装の都合で別名の入力fieldを
推測してはならない。

- `source.year`
- `source.relative_path`
- `kyoku.bakaze`
- `kyoku.kyoku`
- `kyoku.honba`
- `kyoku.oya`
- `kyoku.scores_at_start`
- `kyoku.dora_marker`
- `kyoku.start_kyoku_line`
- `riichi.actor`
- `riichi.riichi_discard_number`
- `riichi.riichi_declaration_tile`
- `riichi.riichi_declaration_tile_kind`
- `riichi.reach_event_index`
- `riichi.declaration_dahai_event_index`
- `riichi.reach_accepted_event_index`
- `riichi.reach_line`
- `riichi.declaration_dahai_line`
- `riichi.reach_accepted_line`
- `hand.concealed_tiles_after_discard`
- `hand.fixed_melds[].meld_type`
- `hand.fixed_melds[].tiles`
- `actor_discards_before_riichi[].discard_number`
- `actor_discards_before_riichi[].tile`
- `actor_discards_before_riichi[].normalized_tile`
- `actor_discards_before_riichi[].tsumogiri`
- `actor_discards_before_riichi[].event_index`
- `actor_discards_before_riichi[].is_riichi_declaration`
- `actor_discards_before_riichi[].was_called`
- `actor_discards_before_riichi[].call_type`
- `actor_discards_before_riichi[].called_by_actor`
- `actor_discards_before_riichi[].call_event_index`
- `waits.wait_tiles`
- `waits.wait_tile_count`
- `waits.wait_details[].tile`
- `waits.wait_details[].hand_type`
- `waits.wait_details[].wait_shape`
- `waits.wait_shapes`
- `waits.contains_ryanmen`
- `waits.is_pure_ryanmen`
- `waits.is_multiwait`

このうちbaselineの主要集計に直接使うのは`source.year`、
`riichi.riichi_discard_number`、`hand`、`waits`である。他のfieldはdataset readerが
schema v1 record全体の整合性を検証するために保持する。

## 観測単位と用語

基本の観測単位は「成立リーチ1件、すなわちdataset 1 record」である。1人が複数局で
リーチした場合も各recordを別観測とし、同一局の複数成立リーチもactorごとに別観測とする。
プレイヤー単位・対局単位の重み付けは行わない。

`wait_tiles`は1 record内で重複しない正規化済み34牌種、`wait_details`は
`tile × hand_type × wait_shape`の関係である。複数待ち牌や同一牌の複数解釈があるため、
`wait_details`の件数をrecord-level比率の分母にしてはならない。

本書では次の3単位を明示的に区別する。

1. record-level: 成立リーチrecordが特性を1つ以上含むか
2. wait-tile-level: 1 record内の正規化待ち牌1種類、すなわち`(record, wait tile)`
3. detail-level: `wait_details`の1要素

## 正式baseline指標

### 共通のcount/rate表現

比率を持つ指標は原則として`count`、`denominator`、`rate`を一組で保存する。
整数countとdenominatorを正本とし、`rate = count / denominator`とする。denominatorが0の
場合は`rate`をJSON `null`、Markdown `N/A`とする。overall rateは全年度のcountと
denominatorから計算し、年度rateの単純平均・加重平均から作らない。

機械可読JSONにはPythonが生成する未丸めの有限floatを保存する。Markdownだけを
百分率小数点以下2桁へ丸め、表示用の値を再計算へ使わない。

### overallと年度別の必須指標

全期間、および2009〜2025の各年について次を保存する。

- `established_riichi_count`
- `pure_ryanmen`: `is_pure_ryanmen`がtrueのcount/rate
- `contains_ryanmen`: `contains_ryanmen`がtrueのcount/rate
- `multiwait`: `is_multiwait`がtrueのcount/rate
- `wait_tile_count_distribution`
- `riichi_discard_number_distribution`

各distributionは値、count、record-level rateを持つ配列とし、整数値の昇順へ並べる。
各distributionのcount合計は`established_riichi_count`と一致しなければならない。
`riichi_discard_number`は1始まりの本人打牌番号をexact valueのまま正本とする。

`is_pure_ryanmen`の意味はdatasetから変更しない。

- 待ち牌種がちょうど2種類
- 全`wait_details`の`hand_type`が`standard`
- 全`wait_details`の`wait_shape`が`ryanmen`
- detailsが空ならfalse

`contains_ryanmen`は、1件以上のdetailが`ryanmen`ならtrueである。記事やMarkdownで
無修飾の「両面率」を使う場合は`is_pure_ryanmen`に基づく純両面率を指すものとし、
同じ表または直後に`contains_ryanmen`率を併記する。両者を混同しない。

`is_multiwait`は`wait_tile_count >= 3`である。

## wait shapeの集計

待ち形をprimary shapeへ潰さず、次の3層を正式出力に含める。

### record-level membership（主要結果）

各recordについて、`wait_details`に現れる`wait_shape`の集合を作る。各shapeごとに
「そのshapeを1つ以上含むrecord」のcount/rateを出す。1 recordが複数shapeへ数えられる
非排他的membershipであるため、shape別rateの合計は必ずしも100%に一致しない。
複数shapeを含むrecordが存在すれば100%を超え得るが、全recordがちょうど1 shapeだけに
属する場合は100%になり得る。

併せて、shape集合そのものの分布`shape_set_distribution`を出す。この分布はrecordごとに
1つだけ数えるため排他的かつ網羅的で、count合計は全record数になる。集合内の順序は
次の仕様順とする。

```text
ryanmen, kanchan, penchan, shanpon, tanki, kokushi_single, kokushi_13men
```

`records_with_multiple_wait_shapes`もrecord-level count/rateとして保存する。

### wait-tile-level interpretations（補助結果）

1 observationを`(record, wait_tiles内の1牌種)`とする。同じ牌種が別recordに現れれば
別observationである。各observationに紐づく全detailからshape集合を作り、次を出す。

- `wait_tile_observation_count = sum(record.wait_tile_count)`
- shape別membership count/rate（非排他的）
- `shape_set_distribution`（排他的、count合計はwait-tile observation数）
- 正規化34牌種ごとのobservation数、shape membership、shape-set分布
- 1待ち牌に複数shape解釈があるobservationのcount/rate

34牌種とshapeは既存datasetの仕様順で出力する。`2m/ryanmen`と`2m/shanpon`を持つ牌を
どちらか一方へ割り当てない。wait-tile-levelのshape membershipも非排他的であり、
全体または牌種内のshape別rate合計は必ずしも100%に一致せず、複数shape解釈があれば
100%を超え得る。

全wait-tile-levelのshape membership rateは`wait_tile_observation_count`を分母とする。
牌種別のshape membership rateとshape-set distribution rateは、その正規化牌種自身の
observation countを分母とする。例えば全wait-tile observationが4件、`3s`が1件で、
その1件がryanmen解釈を持つ場合、`3s`内のryanmen membership rateは`1 / 1`である。
各牌種の出現頻度を示す`tile_frequency`のcount/shareだけは、全体の
`wait_tile_observation_count`を分母とし、牌種内の解釈率と混同しない。

### detail-level（診断結果）

`wait_details`の総件数と、`hand_type × wait_shape`ごとの件数・detail shareを出す。
このshareの分母はdetail総件数であり、成立リーチ件数ではない。field名も
`share_of_details`とし、「リーチの待ち形率」やrecord-level rateとは表現しない。
同一`tile / hand_type / wait_shape`はdataset内で既に重複排除されているため、集計器が
追加のdeduplicationや解釈の削除を行ってはならない。

## hand typeの集計

`hand_type`は`standard`、`chiitoitsu`、`kokushi`の仕様順とする。各recordの全detailから
hand type集合を作り、次をrecord-levelで出す。

- hand type別membership count/rate（非排他的）
- `hand_type_set_distribution`（排他的かつ網羅的）
- `records_with_multiple_hand_types` count/rate

同じrecordまたは同じ待ち牌に`standard`と`chiitoitsu`の両解釈が存在する場合、両方の
membershipへ数える。排他的な「主hand type」は作らない。detail-levelの
`hand_type × wait_shape`表は前節の診断結果を正本とする。

## 年度・巡目集計

年度別結果は2009から2025の昇順で保存し、各年度に正式baseline指標とwait shape・
hand type集計を保持する。

巡目の正本は`riichi.riichi_discard_number`のexact valueである。全期間の各打牌番号、
および年度×打牌番号について次を出す。

- `established_riichi_count`
- `pure_ryanmen` count/rate
- `contains_ryanmen` count/rate
- `multiwait` count/rate
- `wait_tile_count_distribution`

`by-turn`出力は全期間のturn配列と、2009〜2025各年のturn配列を保持する。これにより
巨大なrecord-level派生datasetを作らず年度×巡目を確認できる。turn配列は整数昇順、
年度配列は年度昇順とする。

1〜3打目、4〜6打目などのグループはv1の機械可読な正式指標に含めない。Markdownで
表示用グループを追加する場合も、exact turn表から導出し、区間定義と「表示用派生値」
であることを明記する。

## 天鳳の5枚目待ちと感度確認

### baselineの正本

canonical datasetの`wait_tiles`と`wait_details`は、天鳳の形式聴牌ルール上のformal wait
を表す。純手牌だけで同牌4枚を使用していない限り、fixed ankan等で自己所有4枚となる
牌種の5枚目待ちも保持する。baselineの正式指標ではこのformal waitを削除しない。

### self-owned-fourの判定

正規化牌種`t`について、宣言打牌直後に次を満たし、かつ`t`が`waits.wait_tiles`に
含まれる場合、その`(record, t)`を`self_owned_four_fifth_tile_formal_wait`とする。

```text
normalized_count(hand.concealed_tiles_after_discard, t)
  + normalized_count(all hand.fixed_melds[].tiles, t) == 4

normalized_count(hand.concealed_tiles_after_discard, t) < 4
```

赤5は通常5へ正規化して数える。後半の条件により、純手牌だけで4枚使用していて
candidateにならないケースと区別する。canonical成立リーチではfixed meldはankanのみだが、
集計定義はschemaの全`fixed_melds`を同じ方法で数え、未知のmeld typeを黙って無視しない。

### 感度指標

overallと年度別で少なくとも次を保存する。

- self-owned-four fifth-tile formal waitを1種類以上含むrecord数・率
- 該当する`(record, wait tile)` observation数
- 該当waitだけで構成されるrecord数・率
- 該当waitと通常waitが共存するrecord数・率

さらに感度確認として、該当wait tileと、そのtileに紐づく全`wait_details`を除いた
`self_ownership_adjusted_wait`をrecordごとに一時構成する。除外対象ではないwait tileに
複数interpretationがある場合は、その全detailをそのまま残す。残存`wait_details`を
adjusted側の正本とし、そこに現れるtile集合を仕様の34牌種順へ並べたものを
`adjusted_wait_tiles`とする。次の値を元recordから流用せず、残存正本だけからすべて
再計算する。

- `adjusted_wait_tiles`
- `adjusted_wait_tile_count`
- `adjusted_is_multiwait`
- `adjusted_contains_ryanmen`
- `adjusted_is_pure_ryanmen`
- `adjusted_wait_shape_membership`
- `adjusted_wait_shape_set_distribution`
- `adjusted_hand_type_membership`
- `adjusted_hand_type_set_distribution`

元recordの`is_multiwait`、`contains_ryanmen`、`is_pure_ryanmen`、元`wait_details`から
作られたshape membership、hand type membershipをadjusted系列へ流用してはならない。
adjusted `is_pure_ryanmen`は、残存detailが空でなく、adjusted待ち牌種がちょうど2種類、
全残存detailが`standard/ryanmen`の場合だけtrueとする。

overall、年度別、年度×exact turn、全期間exact turnの各sliceで、次のformal/adjusted
record-level指標を同じrecord母集団について保存する。

- record count
- `wait_tile_count` distribution
- `is_multiwait` count/rate
- `contains_ryanmen` count/rate
- `is_pure_ryanmen` count/rate
- wait shape membership count/rateとexact shape-set distribution
- hand type membership count/rateとexact hand-type-set distribution
- adjusted系列の`adjusted_zero_wait_count`

overallと年度別では主要booleanのformal baselineとの差をcount差とpercentage pointでも
保存する。adjusted側では診断用detail-level cross tableや34牌種別wait-tile-level表を
複製しない。必要な解釈はrecord-level shape/hand type membershipとset distributionで
保持し、formal側の3層集計とadjusted側の感度指標を区別する。

比較可能性を保つため、adjusted主要率の分母も元と同じ全成立リーチrecord数とする。
該当waitを除くと待ちが0種類になるrecordも除外せず、adjusted distributionの
`wait_tile_count=0`へ数え、正式指標`adjusted_zero_wait_count`へ加算する。この場合、
`adjusted_is_multiwait`、`adjusted_contains_ryanmen`、`adjusted_is_pure_ryanmen`はfalse、
adjusted wait shape setとhand type setは空、全shape membershipと全hand type membershipは
falseとする。空集合もset distributionの正規categoryとして1 recordを計上する。

record-level、wait-tile-level、adjustedを含むすべてのshape-setおよびhand-type-set
distributionは、文字列化したsetをkeyにせず、次のように構造化した配列で保存する。
`members`はそれぞれの仕様順に並べ、空集合は`"members": []`で一意に表す。category配列
自体も、空集合を先頭、その後をmembersの仕様順による辞書式順序にして決定論的に保存する。

```json
{
  "members": [],
  "count": 1,
  "denominator": 100,
  "rate": 0.01
}
```

このadjustmentは自己所有情報だけを使う感度確認であり、他家手牌、河、山残枚数、
フリテンを考慮しない。したがって`live_wait`、実効枚数、和了可能率とは呼ばない。
formal baselineの値とadjusted値を置換・混在させない。

## 出力

Git管理する小さな結果だけを`research/results/riichi-wait-summary-v1/`へ置く。
record-level派生datasetは作らず、元の`data/processed/riichi-waits-v1`を正本とする。
正式成果物のlogical filenameは次とする。

```text
riichi-wait-summary-v1-overall.json
riichi-wait-summary-v1-yearly.json
riichi-wait-summary-v1-by-turn.json
riichi-wait-summary-v1.md
```

### 共通metadata

3つのJSONは自己完結した同一のmetadata envelopeを持つ。

- analysis resultの`schema_version`
- analysis nameと観測単位
- input `dataset_name`とdataset `schema_version`
- input manifestのproject-relative pathとSHA256
- dataset generator commit
- analysis codeのgit commitと`worktree_clean=true`
- source repository/release、years、scope
- formal wait semanticsとheadline両面率の定義

timestamp、elapsed、絶対ローカルpathはcanonical resultへ含めない。同一input manifestと
同一analysis commitから同じbytesを再生成できることを優先する。

canonical分析開始前に、analysis codeのGit worktreeについてtracked変更とuntracked fileを
検査し、いずれかが存在すれば拒否する。`git status --porcelain=v1 --untracked-files=all`
相当を明示し、`status.showUntrackedFiles=no`等のユーザー設定で未追跡fileの検出を無効化
できないようにする。.gitignore対象fileはdirtyとはみなさない。このguardにより、記録する
analysis commitが実際に実行したtracked分析コードを含むHEADであることを保証する。
canonical inputでは`source.repository=NikkeTryHard/tenhou-to-mjai`も固定値照合する。

metadata envelopeはanalysis schema v1の一部であり、生成時とpublished generationの読込時に
同じvalidatorで検証する。必須fieldと値は次のとおりとする。必須fieldがないmetadata、型が
異なるmetadata、未知のanalysis name/versionは受理しない。

- `schema_version`は整数1、`analysis_name`は`riichi-wait-summary-v1`
- `observation_unit`は`established_riichi_record`
- `years`は2009〜2025の範囲にある重複なし・昇順の整数配列
- `scope`は`rule_code=00a9`、`aka_flag=true`、`bakaze=E`
- `input_dataset.dataset_name`は`riichi-waits-v1`、同`schema_version`は整数1
- `input_dataset.manifest_path`はproject-relativeなPOSIX path
- `input_dataset.manifest_sha256`は64桁の小文字16進SHA256
- `input_dataset.source_repository`は`NikkeTryHard/tenhou-to-mjai`、
  `input_dataset.source_release_tag`は`v2.0.0`
- `input_dataset.generator_git_commit`と`analysis_generator.git_commit`は空でなく、前後に
  空白のない文字列。Git object formatを40桁へ固定せず、実際にmanifestまたは
  `git rev-parse HEAD`から得た値を保持する
- `input_dataset.totals.output_records`は非負整数。生成時はaggregateのrecord数と一致する
- `analysis_generator.worktree_clean`はboolean `true`
- `formal_wait_semantics`と`ryanmen_metrics`は本仕様で固定したformal waitおよびheadline定義

3つのJSONについて、metadataが相互に同一であることと、そのmetadata自体が上記契約を満たす
ことを別々の条件として検証する。`SummaryAnalysis`の直接構築時にも検証し、文書生成時にも
再検証するため、構築後にmutableなmappingを変更して検証を迂回できないようにする。

### overall JSON

全期間の正式指標、3層のwait shape集計、hand type集計、5枚目待ち感度指標を保持する。
`formal_baseline`と`fifth_tile_sensitivity.adjusted_record_level`は同じrecord母集団を持ち、
主要なrecord-level指標を同じ構造で比較できるようにする。主要構造は次のとおりとする。

```text
metadata
result:
  established_riichi_count
  formal_baseline:
    pure_ryanmen
    contains_ryanmen
    multiwait
    wait_tile_count_distribution
    wait_shape_record_level:
      wait_shape_membership
      wait_shape_set_distribution
      records_with_multiple_wait_shapes
    hand_type_record_level:
      hand_type_membership
      hand_type_set_distribution
      records_with_multiple_hand_types
  riichi_discard_number_distribution
  wait_tile_level                  # formal側だけの補助結果
  wait_detail_level                # formal側だけの診断結果
  fifth_tile_sensitivity:
    affected_record/tile counts
    adjusted_record_level:
      adjusted_zero_wait_count
      adjusted_is_pure_ryanmen
      adjusted_contains_ryanmen
      adjusted_is_multiwait
      adjusted_wait_tile_count_distribution
      adjusted_wait_shape_record_level:
        adjusted_wait_shape_membership
        adjusted_wait_shape_set_distribution
        adjusted_records_with_multiple_wait_shapes
      adjusted_hand_type_record_level:
        adjusted_hand_type_membership
        adjusted_hand_type_set_distribution
        adjusted_records_with_multiple_hand_types
    formal_adjusted_deltas
```

`adjusted_record_level`は前節の`adjusted_*`再計算結果だけから構築し、formal値を参照して
埋めてはならない。`established_riichi_count`と各rateのdenominatorはformal/adjustedで
同一とする。

### yearly JSON

`years`配列に2009〜2025を昇順で格納し、各要素は`year`とoverall相当の年度内
`formal_baseline`、`fifth_tile_sensitivity.adjusted_record_level`、主要deltaを保持する。
formal側のwait-tile-levelおよびdetail-level集計も年度ごとに保持する。overall値を
年度rateの平均から逆算しない。

### by-turn JSON

次の構造でexact turnの主要指標を保持する。

```text
metadata
all_years:
  - riichi_discard_number
  - established_riichi_count
  - formal_baseline record-level summary
  - adjusted_record_level summary including adjusted_zero_wait_count
by_year:
  - year
  - turns: [same formal/adjusted summaries in ascending riichi_discard_number]
```

by-turnではrecord count、wait-tile-count distribution、3主要boolean、shape membership、
hand type membership、exact shape/hand-type set distribution、
`adjusted_zero_wait_count`をformal/adjustedで比較可能にする。34牌種別wait-tile-level表、
detail-level cross table、主要deltaはby-turnへ重複保存しない。

### Markdown

Markdownは同じin-memory aggregateから生成する人間向けsummaryであり、JSONとは独立に
再集計しない。少なくともdataset identity、全体headline、年度表、exact turn表、
wait shapeが非排他的である注意、hand typeの注意、5枚目待ち感度結果、validation結果を
含める。巨大になる場合は年度×turn全表をJSONだけに置き、Markdownには全期間turn表と
年度headlineを載せる。

### 決定論的serialization

- JSONはUTF-8、`sort_keys=true`、`allow_nan=false`、indent 2、LF、末尾LFとする
- 配列は本仕様の年度、整数、34牌種、hand type、wait shape順に従う
- 集合やCounterのiteration orderへ依存しない
- Markdownの行順もJSONと同じ仕様順にする
- JSONの有限floatは丸めず、Markdownだけ表示用に丸める

### publication stateとfailure atomicity

4成果物を個別の固定pathへ順次置換してrollbackする方式は用いない。出力rootは次の
content-addressed generation構造を持つ。

```text
research/results/riichi-wait-summary-v1/
  current.json
  generations/
    <generation_id>/
      riichi-wait-summary-v1-overall.json
      riichi-wait-summary-v1-yearly.json
      riichi-wait-summary-v1-by-turn.json
      riichi-wait-summary-v1.md
      complete.json
```

`generation_id`は4成果物のlogical filename、byte size、SHA256を仕様順に並べたdescriptorの
SHA256とする。同じ入力manifestとanalysis commitから生成した同じ4成果物は同じIDになる。
Windowsでnon-empty directoryを原子的に置換できることには依存しない。

生成時はgeneration directory内で4成果物を一時fileから個別に置換し、全fileを再読込して
size、SHA256、JSON decode、3 JSON間のmetadata一致、および前節のmetadata契約を確認する。
その後、descriptorを持つ`complete.json`を一時fileから置換する。`complete.json`のdescriptor
は各成果物のSHA256を通じて、成果物内のcanonical manifest SHA256やgenerator commitを含む
検証済みmetadataをそのgenerationへ結び付ける。重要なprovenanceを`current.json`だけには
置かない。4成果物完成前や検証失敗時には`complete.json`が存在しないため、そのgenerationは
正式成果物ではない。

publicationの最後に、generation IDとcompletion markerの相対pathだけを持つ
`current.json`を一時fileから`os.replace()`で原子的に更新する。正式成果物とは、
`current.json`が指すgenerationについて、`complete.json`と全4fileのsize/SHA256を検証できる
場合だけをいう。directory内の他generation、completion markerのない途中generation、
`.tmp`残骸は正式成果物として認識しない。

新runの失敗またはprocess終了が`current.json`更新より前なら旧pointerを維持し、更新後なら
検証済み新generation全体が正式となる。新旧4fileが1つの正式世代として混在する状態は
作らない。cleanupはpublication成立条件ではなく、cleanup失敗で既存の正式世代を変更しない。
既存generationは自動削除せず、同じgenerationを再生成する場合はmarkerと全fileを検証して
再利用する。

## streaming設計

既存`load_manifest()`でcompleted manifestを検証し、その`years`配列に記載された
2009〜2025のentryだけを順に処理する。directory内のgzipをglobで自動発見して入力へ
混ぜない。各entryの`output_filename`、実file size、SHA256を既存dataset validationと
同じ条件で検証した後、そのannual gzip JSONLを1回だけ解凍streamingする。
size/SHA256のcompressed-byte検査でfileを読む処理はあり得るが、10,706,714件のJSONL
decode、DTO validation、aggregationは同一の1 passだけで行う。

各JSONL行を既存`iter_dataset_records()`相当の経路でdecodeし、
`RiichiWaitDatasetRecord`構築による全schema・nested invariant validationを行う。
同じpassでrecordをaggregateし、年度末に実読込行数をmanifest entryの
`output_records`と照合する。既存`iter_validated_year_records()`を利用または最小限
共通化すれば、annual size/SHA、record year、DTO、行数の検証を維持したまま実現できる。
集計前に`validate_dataset_integrity(deep=True)`で全recordを別途decodeしてはならない。

validated recordごとに次のbounded aggregateだけを更新する。

- overall accumulator 1個
- 年度accumulator 17個
- exact turn accumulator
- 年度×exact turn accumulator
- shape、hand type、wait tile、detail、5枚目待ち用Counter

record本体や年度全件をlistへ保持しない。処理済みrecordへの参照も保持しない。
1 recordの全levelと全sliceを同じpassで更新する。年次fileの切替時に年度countをmanifestと
照合し、全年度後にglobal count 10,706,714、manifest totals、distribution不変条件を
照合する。途中のrecord validation、年度count、最終count、集計不変条件のいずれかが
失敗した場合、一時出力を正式成果物としてpublishしない。

進捗表示は一定record数または一定file内行数ごとにyear、processed annual records、
overall records、elapsedを標準エラーへ出してよいが、出力結果へelapsedを混ぜない。

## validation方針

### 人工recordテスト

production fixtureの巨大な実牌譜は再生せず、schema v1の最小recordまたは集計入力用の
不変DTOで次を独立に固定する。

- 2種類・全standard/ryanmenのpure ryanmen
- ryanmenを含むがpureでない複合待ち
- ryanmenを含まない待ち
- `wait_tile_count >= 3`のmultiwait
- chiitoitsu
- kokushi singleとkokushi 13men
- 同一wait tileに複数shape interpretationがあるrecord
- standardとchiitoitsu等、複数hand type interpretationがあるrecord
- fixed ankanによるself-owned-four fifth-tile formal waitだけのrecord
- fifth-tile formal waitと通常waitが共存するrecord
- 純手牌4枚のためformal candidateに含まれないrecord
- `5mr`、`5pr`、`5sr`を通常5へ合算するself-owned-four判定
- 複数turnと複数yearの集計
- 入力順を変えても同じcanonical output順になること

5枚目待ち感度分析では抽出仕様R17〜R19を再利用し、summary-analysis側のfixture名を
`summary-R17`、`summary-R18`、`summary-R19`として次の遷移を完全一致で固定する。

| Fixture | formal wait | adjusted wait | 必須期待値 |
|---|---|---|---|
| `summary-R17` | `3s` | 空 | recordを維持し、`adjusted_wait_tile_count=0`、`adjusted_zero_wait_count`へ1、shape setとhand type setは空、全membershipと3主要booleanはfalse |
| `summary-R18` | `2m, 5m, 8m` | `5m, 8m` | pure ryanmenはfalse→true、multiwaitはtrue→false、contains ryanmenはtrueを維持 |
| `summary-R19` | `2m, 5m` | `2m` | pure ryanmenはtrue→false、contains ryanmenはtrueを維持、multiwaitはfalseを維持 |

`summary-R18`と`summary-R19`では、残存detailからshape membershipが`ryanmen`のみ、
shape setが`["ryanmen"]`、hand type membershipが`standard`のみ、hand type setが
`["standard"]`になることも検査する。`summary-R17`では両set distributionの
`"members": []` categoryへ1件を数える。除外されないwait tileに複数shapeまたは
複数hand type解釈を持たせたfixtureも用意し、その全解釈がadjusted側へ残ることを
完全一致で検査する。

各ケースではrecord-level membership、wait-tile-level shape set、detail-level件数、
hand type集合、主要boolean、distribution、5枚目感度のcountとdenominatorまで完全一致を
確認する。set化によって同一tileの複数interpretation欠落を隠さない。

### 集計不変条件テスト

- `pure_ryanmen.count <= contains_ryanmen.count <= established_riichi_count`
- 各排他的distributionのcount合計がそのlevelのdenominatorと一致
- 非排他的membershipは合計100%を要求しない
- `wait_tile_observation_count == sum(wait_tile_count)`
- detail cross tableの合計がdetail総件数と一致
- 年度count合計がoverall countと一致
- turn count合計がoverall countと一致
- 年度×turn countの年度内合計が各年度countと一致
- formal fifth-tile対象recordが「fifth-only」と「fifth + other」に分割される
- adjusted waitが空のrecordをdropしない
- adjustedとformalのrecord countおよびrate denominatorが各sliceで一致する
- adjusted shape/hand type set distributionは空集合categoryを含めてrecord countへ合計する
- adjusted membershipと3主要booleanは残存detailsからの再計算値と一致する
- zero denominatorのrateが`null`になる
- 不正boolean、未知shape/hand type、非正規化wait tile、不正JSONを拒否する

### stream・outputテスト

- 複数annual gzipを順にstreamし、全件をlist化しない
- manifest年度順とannual record countを検証する
- annual count mismatchとglobal count mismatchをfail closedで拒否する
- canonical profileでは10,706,714件との完全一致を要求する
- 同じfixtureを2回集計したJSON/Markdownがbyte一致する
- JSON 3種とMarkdownが同じmetadata fingerprintを持つ
- overall、yearly、by-turnのformal/adjusted主要指標が同じschemaとdenominatorを持つ
- outputは全処理成功前にfinalとしてpublishされない
- malformedまたは不正な`current.json`、存在しない・未完成generationへのpointerを拒否する
- `complete.json`の必須field、generation ID、4 artifact descriptorを検証する
- published artifactの欠損、size不一致、同一sizeでのSHA256不一致を拒否する
- 空または不正なmetadata、および3 JSON間で異なるmetadataを拒否する
- 同一generation IDの未完成残骸は安全に再生成し、完成世代は全fileを再検証して再利用する
- 同一generation IDの完成世代が破損していれば再利用せず、silent successしない
- 新世代のpublication失敗時も、既存`current.json`が指す旧完成世代を維持する

### 本番結果の確認

本番runでは最低限、次を実行記録へ残す。

- completed manifestと全17annual artifactのsize/SHA256検証成功
- 全annual JSONLの1-pass deep record validationとaggregation成功
- 年度別read countとmanifest一致
- total read count 10,706,714
- 全集計不変条件の成功
- input manifest SHA256とanalysis commit
- JSONから再計算した主要count/rateとMarkdown表示の一致

独立した小規模reducerまたはSQL等による主要headlineの再集計をreview時に行うことが
望ましい。一致は信頼性を高めるが、分析の絶対的正しさを証明するものではない。

## 実装時の責務分離と予定file

本仕様の実装時は、dataset schema validation、1 recordの集計、複数sliceのmerge、
serialization、CLIを分離する。麻雀の待ち探索やMJAI replayは再実装しない。

主な予定fileは次とする。

```text
src/mahjong_analysis/riichi_wait_summary.py
analysis/summarize_riichi_waits.py
tests/test_riichi_wait_summary.py
docs/specs/riichi-wait-summary-analysis.md
research/results/riichi-wait-summary-v1/current.json
research/results/riichi-wait-summary-v1/generations/<generation_id>/
```

実装段階で既存`riichi_wait_dataset.py`に変更が必要な場合も、集計から独立した汎用reader
またはmanifest validationの最小共通化に限定する。canonical dataset artifactは変更しない。

## 未決事項と後続範囲

次はbaseline v1の実装を妨げない後続判断とし、今回の正式指標へ混ぜない。

- dealer/child、actor、局番号、点数状況によるsubgroupを追加するか
- exact turnからどの表示用turn groupを派生させるか
- 年度差の信頼区間、heterogeneity、trend testを別分析として行うか
- 対子落とし・カンチャン落としに見える河の操作的定義
- 物理的な残枚数、他家情報、フリテンを含むlive-wait指標
- Parquet等の分析用派生形式を別途作るか

baseline v1ではrecord-levelの`is_pure_ryanmen`率をheadlineの両面率、
`contains_ryanmen`率を包含的な補助指標とする。wait shapeを排他的分類へ潰さないこと、
exact turnを正本とすること、formal waitとself-ownership adjustmentを分離することは
確定仕様である。
