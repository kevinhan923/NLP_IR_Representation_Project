
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import ir_datasets


def warm_ir_dataset(name: str, touch_docs: bool = True, touch_queries: bool = True, touch_qrels: bool = True, max_touch: int = 5) -> None:
    print(f"[download] warming ir_datasets dataset: {name}")
    ds = ir_datasets.load(name)
    if touch_docs and hasattr(ds, 'docs_iter'):
        try:
            it = ds.docs_iter()
            for idx, _ in enumerate(it):
                if idx + 1 >= max_touch:
                    break
        except Exception as e:
            print(f"[warn] docs warm-up failed for {name}: {e}")
    if touch_queries and hasattr(ds, 'queries_iter'):
        try:
            it = ds.queries_iter()
            for idx, _ in enumerate(it):
                if idx + 1 >= max_touch:
                    break
        except Exception as e:
            print(f"[warn] queries warm-up failed for {name}: {e}")
    if touch_qrels and hasattr(ds, 'qrels_iter'):
        try:
            it = ds.qrels_iter()
            for idx, _ in enumerate(it):
                if idx + 1 >= max_touch:
                    break
        except Exception as e:
            print(f"[warn] qrels warm-up failed for {name}: {e}")


def download_beir_dataset(name: str, out_dir: Path) -> None:
    url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip'
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f'{name}.zip'
    final_dir = out_dir / name
    if final_dir.exists() and (final_dir / 'corpus.jsonl').exists():
        print(f'[skip] BEIR dataset already present: {final_dir}')
        return
    print(f'[download] downloading BEIR dataset: {name}')
    urllib.request.urlretrieve(url, zip_path)
    print(f'[download] extracting: {zip_path.name}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    zip_path.unlink(missing_ok=True)
    # Some archives extract into a same-named folder, but keep a small safety pass.
    nested = out_dir / name / name
    if nested.exists() and (nested / 'corpus.jsonl').exists():
        tmp = out_dir / f'{name}_tmp_flatten'
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.move(str(nested), str(tmp))
        shutil.rmtree(out_dir / name)
        shutil.move(str(tmp), str(out_dir / name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data')
    parser.add_argument('--download-msmarco', action='store_true')
    parser.add_argument('--download-beir', action='store_true')
    parser.add_argument('--msmarco-doc-id', default='msmarco-document/trec-dl-2019')
    parser.add_argument('--beir-datasets', nargs='*', default=['nfcorpus', 'scifact', 'fiqa'])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.download_msmarco:
        warm_ir_dataset(args.msmarco_doc_id)

    if args.download_beir:
        beir_root = root / 'beir'
        for name in args.beir_datasets:
            download_beir_dataset(name, beir_root)

    print('[done] data download / warm-up step finished.')
    print('[note] ir_datasets caches downloaded data in its own cache directory.')


if __name__ == '__main__':
    main()
