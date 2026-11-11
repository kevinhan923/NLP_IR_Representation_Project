# 未入库的大文件

下面 4 个文件（共 1.51 GB，其中 4 个超过 100 MB）**不在这个 git 仓库里**，从 GitHub 克隆下来时这些路径是空的。
原因：GitHub 单个文件超过 100 MB 会被拒收。四个语料文件都超过 GitHub 的单文件上限；根目录 `.gitignore` 的 `data/` 规则排除了整个数据目录。

## 文件在哪

- 真文件：`~/Documents/GitHub Backup/NLP_IR_Representation_Project/<下表中的路径>`，由 iCloud 备份。
- 本机仓库里：下表每个路径都是指向真文件的软链接（被 `.gitignore` 忽略），所以代码按原路径读取即可。
- 总规则见 `~/Github/CLAUDE.md`；新增或删除这类文件时要同步更新本文件和 `~/Documents/GitHub Backup/CLAUDE.md`。

## 清单

| 路径（相对仓库根） | 大小 | 是什么 / 丢了怎么办 | SHA-256 |
|---|---|---|---|
| `v1/data/beir/msmarco-doc/corpus.jsonl` | 388.3 MB | BEIR 公开语料，可重新下载 | `7eb819a96d0214e00c823feea182f78b301d188903589d85b1e29e82b7911f0e` |
| `v1/data/beir/scidocs/corpus.jsonl` | 245.4 MB | BEIR 公开语料，可重新下载 | `328bb38854179c83ee40d4f49fae2ef7209ede7163785193cbdcd5906b28eecc` |
| `v1/data/beir/trec-covid/corpus.jsonl` | 211.1 MB | BEIR 公开语料，可重新下载 | `aded69896598665082316d00df487d22212cc5a8611db05db34eb40d04ed00d7` |
| `v1/data/beir/webis-touche2020/corpus.jsonl` | 702.2 MB | BEIR 公开语料，可重新下载 | `52e5585b5c23164d37cf9a59cf5d3f55645e0efeec8cd8cdb6b0bd37e48ec111` |

## 在新克隆（或换电脑）后恢复

先让 iCloud 把 `GitHub Backup` 同步下来，然后在仓库根目录执行，给 Backup 里的每个文件在原位置建软链接：

```bash
B="$HOME/Documents/GitHub Backup/NLP_IR_Representation_Project"
(cd "$B" && find . -type f ! -name .DS_Store) | sed 's|^\./||' | while IFS= read -r f; do
  mkdir -p "$(dirname "$f")"
  [ -e "$f" ] || [ -L "$f" ] || ln -s "$B/$f" "$f"
done
git status --porcelain   # 应该没有输出
```

核对某个文件是否完好：`shasum -a 256 "<路径>"`，与上表的 SHA-256 比较。
