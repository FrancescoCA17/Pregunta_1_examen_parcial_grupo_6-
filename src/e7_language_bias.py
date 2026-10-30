"""Entregable 7: sesgo lingüístico y geográfico de las etiquetas débiles.

Cuantifica, por idioma del reporte (proxy del centro de origen):
  1. Representación: estudios en la población, en la muestra de entrenamiento y entre los
     58 dorados (¿con cuántos casos se puede validar cada idioma?).
  2. Calidad de la extracción: acuerdo con los dorados (U-ones, pares estudio×etiqueta),
     con IC 95 % por bootstrap de estudios; además inglés frente al resto (diferencia pareada
     por remuestreo independiente de cada grupo).
  3. Estilo de reporte en la población no dorada: longitud del texto, menciones inciertas,
     evidencia débil y tasa de positivos de hallazgos "blandos" (derrame, sinovitis,
     contusión). Diferencias aquí cambian la etiqueta débil aunque la imagen sea igual.
  4. Dominio: fabricante del resonador por idioma en la muestra (idioma ≈ centro ≈ equipo).

Salidas: reports/e7_language.csv, e7_reporting_style.csv, e7_vendor_by_language.csv,
e7_summary.txt
"""
import numpy as np
import pandas as pd

from common import LABELS, META, ROOT, SEED

OUT = ROOT / "reports"
N_BOOT = 1000
SOFT = ["Effusion", "Synovitis", "Contusion"]
POS = {"pos_concl", "pos_body", "pos_weak"}


def boot_acc(g, p, rng):
    """Exactitud por pares y su IC remuestreando estudios (filas)."""
    n = len(g)
    acc = (g == p).mean()
    bs = [(g[i] == p[i]).mean() for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return acc, *np.percentile(bs, [2.5, 97.5]), np.array(bs)


def main():
    rng = np.random.default_rng(SEED)
    cat = pd.read_csv(META / "study_catalog.csv")
    weak = pd.read_csv(META / "weak_labels.csv")
    train = pd.read_csv(META / "train.csv")
    sample = pd.read_csv(META / "sample_split.csv")[["StudyInstanceUID", "split"]]
    df = cat.merge(weak, on="StudyInstanceUID").merge(
        train[["StudyInstanceUID", "Report"] + LABELS].rename(columns={L: f"{L}_gold" for L in LABELS}),
        on="StudyInstanceUID").merge(sample, on="StudyInstanceUID", how="left")

    # 1-2. representación y acuerdo con dorados
    gold = df[df.gold]
    rows, boots = [], {}
    for lg, g in df.groupby("lang"):
        r = dict(lang=lg, poblacion=len(g), pct_poblacion=100 * len(g) / len(df),
                 muestra=int(g.split.isin(["weak_train", "weak_val"]).sum()),
                 dorados=int(g.gold.sum()))
        gg = gold[gold.lang == lg]
        if len(gg):
            a, lo, hi, bs = boot_acc(gg[[f"{L}_gold" for L in LABELS]].values,
                                     gg[LABELS].values, rng)
            r.update(acuerdo=a, acuerdo_lo=lo, acuerdo_hi=hi)
        rows.append(r)
    lang = pd.DataFrame(rows).sort_values("poblacion", ascending=False)

    en, rest = gold[gold.lang == "en"], gold[gold.lang != "en"]
    a_en, *_, b_en = boot_acc(en[[f"{L}_gold" for L in LABELS]].values, en[LABELS].values, rng)
    a_rs, *_, b_rs = boot_acc(rest[[f"{L}_gold" for L in LABELS]].values, rest[LABELS].values, rng)
    d_lo, d_hi = np.percentile(b_en - b_rs, [2.5, 97.5])

    # 3. estilo de reporte (población no dorada)
    pop = df[~df.gold].copy()
    pop["largo"] = pop.Report.fillna("").str.len()
    st = pop[[f"{L}_status" for L in LABELS]]
    style = pd.DataFrame({
        "estudios": pop.groupby("lang").size(),
        "largo_mediano": pop.groupby("lang").largo.median(),
        "pct_reporte_corto": pop.assign(c=pop.largo < 300).groupby("lang").c.mean() * 100,
        "pct_incierto": st.eq("uncertain").mean(axis=1).groupby(pop.lang).mean() * 100,
        "pct_evidencia_debil": st.eq("pos_weak").mean(axis=1).groupby(pop.lang).mean() * 100,
    })
    for L in SOFT:
        style[f"pos_{L}"] = pop[f"{L}_status"].isin(POS).groupby(pop.lang).mean() * 100
    style = style.sort_values("estudios", ascending=False)

    # 4. fabricante por idioma (muestra con volumen sagital)
    idx = pd.read_csv(ROOT / "data" / "processed" / "sag_index.csv")[["StudyInstanceUID", "manufacturer"]]
    m = idx.merge(cat, on="StudyInstanceUID")
    m["fabricante"] = m.manufacturer.str.upper().str.extract(r"(GE|SIEMENS|PHILIPS|TOSHIBA|CANON|HITACHI)")[0].fillna("OTRO")
    vendor = (pd.crosstab(m.lang, m.fabricante, normalize="index") * 100).round(0)
    vendor["n"] = m.groupby("lang").size()

    OUT.mkdir(exist_ok=True)
    lang.round(3).to_csv(OUT / "e7_language.csv", index=False)
    style.round(1).to_csv(OUT / "e7_reporting_style.csv")
    vendor.to_csv(OUT / "e7_vendor_by_language.csv")
    txt = "\n".join([
        "=== Representación y acuerdo con los dorados (U-ones, pares estudio×etiqueta; IC bootstrap)",
        lang.round(3).to_string(index=False),
        f"\nInglés {a_en:.3f} (n={len(en)}) vs resto {a_rs:.3f} (n={len(rest)}): "
        f"diferencia {a_en - a_rs:+.3f} [IC 95 % {d_lo:+.3f}; {d_hi:+.3f}]",
        "\n=== Estilo de reporte en la población no dorada (%)",
        style.round(1).to_string(),
        "\n=== Fabricante por idioma en la muestra (% de estudios)",
        vendor.to_string(),
    ])
    (OUT / "e7_summary.txt").write_text(txt, encoding="utf8")
    print(txt)


if __name__ == "__main__":
    main()
