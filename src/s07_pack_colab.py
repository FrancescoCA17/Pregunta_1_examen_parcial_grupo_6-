"""Paso 7: empaqueta código + datos procesados en un solo archivo para Colab.

colab_bundle.zip (≈1,7 GB) contiene src/, requirements.txt, data/meta/sample_split.csv
y data/processed/{sag,cor} con sus índices. Se sube UNA vez a Google Drive
(MyDrive/knee/) y el cuaderno notebooks/train_colab.ipynb lo copia al disco local
de Colab. Los .npz ya están comprimidos, así que se guardan sin recomprimir.
colab_code.zip (pocos KB) contiene solo src/: si cambia el código basta subir este
archivo a MyDrive/knee/; el cuaderno lo descomprime encima del paquete grande.
Es material derivado de Kaggle: mantener la carpeta de Drive privada.
"""
import zipfile

from common import META, ROOT

OUT = ROOT / "colab_bundle.zip"
CODE = ROOT / "colab_code.zip"


def main():
    files = sorted((ROOT / "src").glob("*.py")) + [ROOT / "requirements.txt", META / "sample_split.csv"]
    for view in ("sag", "cor"):
        files.append(ROOT / "data" / "processed" / f"{view}_index.csv")
        files += sorted((ROOT / "data" / "processed" / view).glob("shard_*"))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"{OUT.name}: {len(files)} archivos, {OUT.stat().st_size / 1e9:.2f} GB")
    code = sorted((ROOT / "src").glob("*.py"))
    with zipfile.ZipFile(CODE, "w", zipfile.ZIP_DEFLATED) as z:
        for f in code:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"{CODE.name}: {len(code)} archivos, {CODE.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
