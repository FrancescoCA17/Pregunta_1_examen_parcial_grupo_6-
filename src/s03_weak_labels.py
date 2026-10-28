"""Paso 3: etiquetas débiles a partir del reporte en texto libre (9 idiomas).

Pipeline por reporte:
  1. Normalización: NFKC, casefold y eliminación de tildes (ménisque -> menisque,
     έσω -> εσω, außen -> aussen). Así un mismo patrón cubre variantes.
  2. Se elimina la sección clínica (indicación/antecedentes), porque menciona
     sospechas ("¿rotura meniscal?") y no hallazgos.
  3. Se separa la conclusión/impresión del resto del cuerpo.
  4. Oraciones -> cláusulas (comas, ':' y conectores adversativos).
  5. Para cada etiqueta: término de hallazgo (+ estructura anatómica cuando hace
     falta) en la misma cláusula o en la anterior de la misma oración.
  6. Cada mención se clasifica en positiva, negada (al estilo NegEx: indicador antes
     o después dentro de la cláusula) o incierta ("posible", "DD", "?").
  7. Agregación con confianza:
       pos en conclusión 0.95 | pos solo en cuerpo 0.85 (0.60 si la conclusión
       lo niega) | evidencia débil 0.70 | incierta 0.50 | negada 0.95 |
       no mencionada 0.80 (0.60 en reportes cortos).

Las reglas se escribieron leyendo solo reportes NO dorados y siguiendo las
definiciones de la pestaña "Data" de Kaggle (menisco = rotura, ligamento =
lesión, OA por compartimento). Los 58 dorados se usan solo en s03b para medir.

Salida: data/meta/weak_labels.csv con, por etiqueta, <L> (0/1, inciertas como 1),
<L>_conf, <L>_p (objetivo blando) y <L>_status; más weak_labels_evidence.csv.
"""
import re
import unicodedata

import pandas as pd

from common import LABELS, META

# ---------------------------------------------------------------- normalización

def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).replace("[DATE]", " ")
    # Saltos de línea por ajuste de página (hr, es): unir salvo fin de oración o viñeta
    text = re.sub(r"(?<![.:;\n])[ \t]*\n(?![ \t]*[\n\-•>*\d])", " ", text)
    text = text.casefold().replace("ı", "i").replace("ς", "σ")
    text = text.replace("v.a.", "verdacht ").replace("d.d.", "dd ")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def rx(*alts):
    """Une alternativas en una regex que empieza en límite de palabra (prefijo).
    Los patrones pasan por la misma normalización de sigma final que el texto."""
    return re.compile(r"(?<!\w)(?:" + "|".join(alts).replace("ς", "σ") + r")")


# ---------------------------------------------------------------- léxico (ya normalizado)

# Secciones
CLIN_HDR = rx(r"clinical (?:history|indication|information|details|question)s?\s*:",
              r"indications?\s*:", r"history\s*:", r"antecedentes(?: clinicos)?\s*:",
              r"motivo(?: de consulta)?\s*:", r"klinische (?:inlichtingen|angaben)\s*:",
              r"diagnostische vraagstel+ing\s*:", r"fragestellung\s*:",
              r"anamne\w*\s*:", r"renseignements cliniques\s*:", r"indication\s*:",
              r"klinik bilgi\w*\s*:", r"endikasyon\s*:", r"uputn\w* dijagnoz\w*\s*:",
              r"клинич\w*\s*\w*\s*:", r"κλινικ\w* (?:ερωτημα|πληροφοριες)\s*:")
BODY_HDR = rx(r"findings\s*:", r"hallazgos\s*:", r"bevindingen\s*:", r"befunde?\s*:",
              r"resultats\s*:", r"resultados\s*:", r"trouvailles\s*:", r"nalaz\s*:",
              r"bulgular\s*:", r"ευρηματα\s*:?", r"мр находка\s*:", r"находка\s*:",
              r"technique\s*:", r"tecnica\s*:", r"technik\s*:", r"texnikh\s*:",
              r"τεχνικη\s*:", r"tetkik protokol\w*\s*:", r"scanprotocol\b",
              r"protocol\w*\s*:", r"knochen\s*:", r"exam performed\s*:")
CONCL_HDR = rx(r"impression\w*\s*\w*\s*:", r"conclusions?\s*:", r"conclusion\s+:",
               r"impresion\w*\s*\w*\s*:", r"conclusion\w*\s*:", r"conclusie\s*:",
               r"besluit\s*:", r"beurteilung\s*:", r"zusammenfassung\s*:",
               r"zakljucak\s*:", r"misljenje\s*:", r"заключение\s*:",
               r"συμπερασμα\s*:?", r"γνωματευση\s*:?", r"sonuc\s*:", r"kanaat\s*:",
               r"yorum\s*:", r"opinion\s*:", r"summary\s*:")

# Negación, incertidumbre
NEG_PRE = rx(r"no\b", r"not\b", r"without\b", r"negative for", r"absence of", r"absent\b",
             r"free of", r"sin\b", r"ni\b", r"ausencia", r"descart\w*", r"niega",
             r"pas d", r"pas de", r"aucun\w*", r"sans\b", r"absence", r"ni\b",
             r"kein\w*", r"ohne\b", r"nicht\b", r"ausschluss", r"geen\b", r"zonder\b",
             r"niet\b", r"nema\b", r"bez\b", r"ne\b", r"nije\b", r"nisu\b", r"nisam\b",
             r"няма\b", r"без\b", r"не\b", r"липса", r"χωρις\b", r"δεν\b", r"ουδεν",
             r"απουσια", r"μη\b", r"yok\w*", r"olmaksizin", r"bulunmam\w*")
NEG_POST = rx(r"not (?:seen|identified|demonstrated|visualized|present|evident)",
              r"(?:is|are|was) not\b", r"ruled out", r"excluded", r"no se (?:observa|identifica|aprecia)",
              r"ausgeschlossen", r"nicht\b",
              r"yok\w*", r"izlenme\w*", r"saptanma\w*", r"gorulme\w*", r"rastlanma\w*",
              r"degil\w*", r"izlenmedi", r"nije\b", r"nisu\b", r"не се\b",
              r"δεν\b", r"n'est pas", r"ne sont pas", r"is uitgesloten")
UNCERTAIN = rx(r"possibl\w*", r"probabl\w*", r"likely", r"suspect\w*", r"suspicious",
               r"cannot be excluded", r"can not be excluded", r"may represent", r"questionable",
               r"equivocal", r"r/o\b", r"rule out", r"to be excluded", r"posibl\w*", r"probabl\w*", r"sospech\w*", r"sugest\w*",
               r"no se descarta", r"dudos\w*", r"possible", r"suspicion", r"evoqu\w*",
               r"moglich\w*", r"verdacht", r"v\.a\.", r"dd\b", r"fraglich\w*", r"eventuel\w*",
               r"mogelijk\w*", r"vermoedelijk", r"waarschijnlijk", r"moguc\w*", r"vjerojatn\w*",
               r"suspekt\w*", r"вероятн\w*", r"възможн\w*", r"съмнени\w*", r"πιθαν\w*",
               r"υποψια", r"ενδεχομεν\w*", r"olasi\w*", r"muhtemel\w*", r"supheli\w*",
               r"\?")

# Términos compartidos
TEAR = rx(r"tear\w*", r"torn\b", r"ruptur\w*", r"rotur\w*", r"roto\b", r"rota\b", r"desgarr\w*",
          r"dechirure\w*", r"rupture\w*", r"\w*riss\w*", r"\w*scheur\w*", r"\w*ruptuur",
          r"\w*ruptur\w*", r"puknu\w*", r"razdor\w*", r"скъсван\w*", r"разкъсван\w*",
          r"руптур\w*", r"ρηξ\w*", r"yirti[kg]\w*", r"yırtık\w*", r"rupture", r"bucket",
          r"flap\b", r"lesion meniscal\w*", r"desinser\w*", r"anse de seau", r"korbhenkel\w*",
          r"grad\w* (?:3|iii)\b", r"grade (?:3|iii)\b", r"stupnja? (?:3|iii)\b")
SPRAIN = rx(r"sprain\w*", r"injur\w*", r"lesion\w*", r"insufficien\w*", r"laxit\w*", r"avuls\w*",
            r"esguince", r"distension", r"elongacion", r"lesion\w*", r"entorse", r"distorsion",
            r"lesion", r"zerrung", r"verletz\w*", r"lasion\w*", r"distorsie", r"verstuiking",
            r"letsel\w*", r"distenzij\w*", r"istegnu\w*", r"ozljed\w*", r"lezij\w*",
            r"разтягане", r"дисторзи\w*", r"увред\w*", r"лезия", r"травм\w* лез\w*",
            r"διαταση", r"θλαση", r"κακωση", r"βλαβη", r"sprain", r"burkul\w*", r"hasar\w*",
            r"lezyon\w*", r"grade (?:1|2|i|ii)\b", r"grad\w* (?:1|2|i|ii)\b")
EDEMA = rx(r"o?edema\w*", r"oedem\w*", r"œdem\w*", r"\w*odem\w*", r"\w*oedeem\w*", r"edem\w*",
           r"едем\w*", r"оток", r"οιδημ\w*")

# Estructuras
MENISCUS = rx(r"menisc\w*", r"menisk\w*", r"menisque\w*", r"meniskus\w*", r"menisku\w*",
              r"\w*meniskus\w*", r"\w*menisk\w*", r"\w*meniscus\w*", r"менискус\w*", r"мениск\w*",
              r"μηνισκ\w*", r"meniskopat\w*", r"meniscopat\w*")
MED_W = rx(r"medial\w*", r"medyal\w*", r"mediaa?l\w*", r"medijaln\w*", r"медиал\w*", r"εσω\b", r"εσωτερικ\w*",
           r"intern\w*", r"inner\b", r"ic\b", r"innen\w*", r"binnen\w*")
LAT_W = rx(r"lateral\w*", r"lateraa?l\w*", r"lateraln\w*", r"латерал\w*", r"εξω\b", r"εξωτερικ\w*",
           r"extern\w*", r"outer\b", r"dis\b", r"aussen\w*", r"buiten\w*")
BOTH_MENISCI = rx(r"both menisci", r"ambos meniscos", r"los meniscos", r"beide[nr]? menisk\w*",
                  r"beide menisci", r"oba menisk\w*", r"двата менискус\w*", r"двата мениск\w*",
                  r"αμφοτερ\w* (?:τ\w* )?μηνισκ\w*", r"her iki menisk\w*", r"deux menisques",
                  r"menisci\b(?! intact)")
OTHER_STRUCT = rx(r"ligament\w*", r"ligamen\w*", r"kruisband\w*", r"kreuzband\w*", r"cruzado\w*",
                  r"croise\w*", r"capraz\w*", r"krizn\w*", r"кръстн\w*", r"χιαστ\w*",
                  r"\w*band\b", r"tendon\w*", r"sehne\w*", r"pees", r"tetiv\w*", r"acl\b",
                  r"mcl\b", r"lca\b", r"cartilag\w*", r"cartilago\w*", r"knorpel\w*")
PATELLA_W = rx(r"\w*patell\w*", r"rotul\w*", r"facet\w*", r"trochle\w*", r"troclea", r"пател\w*",
               r"επιγονατιδ\w*", r"retropatell\w*")

ACL = rx(r"acl\b", r"lca\b", r"vkb\b", r"\w*anterior cruciate", r"cruzado anterior",
         r"cruzados anterior", r"croise anterieur", r"vorder\w* kreuzband\w*",
         r"voorste kruisband\w*", r"prednj\w* krizn\w*", r"предн\w* кръстн\w*",
         r"προσθι\w* χιαστ\w*", r"on capraz\w*", r"anterior capraz\w*", r"ön çapraz\w*",
         r"lcae\b")
MCL = rx(r"mcl\b", r"lcm\b", r"lci\b", r"medial collateral", r"tibial collateral",
         r"colateral (?:medial|interno|tibial)", r"collateral (?:interne|medial)",
         r"innenband\w*", r"medial\w* kollateralband\w*", r"mediale? collatera\w*",
         r"mediaal collateraal", r"medijaln\w* kolateraln\w*", r"медиалн\w* колатерал\w*",
         r"εσω πλαγι\w*", r"medial kollateral\w*", r"medyal kollateral\w*", r"ic yan bag\w*",
         r"collateral (?:ligament )?interne")

OA = rx(r"osteoarthr\w*", r"\w*arthros\w*", r"\w*artros\w*", r"\w*arthrose\w*", r"\w*artroz\w*",
        r"гонартроз\w*", r"остеоартр\w*", r"артроз\w*", r"οστεοαρθρ\w*", r"αρθροπαθ\w*",
        r"osteoartrit\w*", r"dejeneratif artrit\w*", r"dejeneratif eklem hastalig\w*",
        r"degenerative (?:joint|arthrit)\w*",
        r"oa\b", r"djd\b", r"gonartr\w*", r"kraakbeenlijden", r"arthropath\w*")
OA_WEAK = rx(r"osteophyt\w*", r"osteofit\w*", r"остеофит\w*", r"οστεοφυτ\w*", r"osteofiet\w*",
             r"full[- ]thickness", r"espesor total", r"pleine epaisseur", r"vollschicht\w*",
             r"pune debljine", r"ολικου παχους", r"tam kat\w*", r"grade (?:3|4|iii|iv)\b",
             r"grado (?:3|4|iii|iv|tres|cuatro)\b", r"graad (?:3|4)\b", r"icrs (?:3|4|iii|iv)\b",
             r"denudac\w*", r"bone on bone", r"(?:3|4|iii|iv)\.? stupnj\w*",
             r"hondromalacij\w* (?:3|4|iii|iv)\b", r"chondromalac\w* (?:grade )?(?:3|4|iii|iv)\b", r"eklem aralig\w* dara\w*", r"joint space narrow\w*",
             r"epaisseur totale", r"degenerative changes?", r"cambios degenerativos",
             r"modifications degeneratives", r"changements degeneratifs",
             r"degenerativ\w* veranderung\w*", r"degeneratiev\w*", r"degenerativn\w* promjen\w*",
             r"дегенеративн\w* промен\w*", r"εκφυλιστικ\w*", r"dejeneratif\w*")
# En cláusulas sobre menisco o ligamentos, "degeneración" o "grado III" no es artrosis
OA_EXCLUDE_CTX = rx(r"menisc\w*", r"menisk\w*", r"menisque\w*", r"\w*meniskus\w*", r"\w*menisk\w*",
                    r"\w*meniscus\w*", r"мениск\w*", r"μηνισκ\w*", r"ligament\w*", r"ligamen\w*",
                    r"kruisband\w*", r"kreuzband\w*", r"cruzado\w*", r"croise\w*", r"capraz\w*",
                    r"krizn\w*", r"кръстн\w*", r"χιαστ\w*", r"tendon\w*", r"sehne\w*", r"pees",
                    r"tetiv\w*", r"сухожил\w*", r"τενοντ\w*", r"tendin\w*", r"acl\b", r"mcl\b")
COMP_PF = rx(r"patellofem\w*", r"patelofem\w*", r"femoropatel\w*", r"femoro-patel\w*",
             r"retropatell\w*", r"patello-?femor\w*", r"пателофем\w*", r"επιγονατιδομηρια\w*",
             r"fp\b", r"pf\b", r"patell\w*", r"rotul\w*", r"trochle\w*", r"troclea\w*",
             r"пател\w*", r"επιγονατιδ\w*")
COMP_ALL = rx(r"tricompart\w*", r"three compartments", r"tres compartimentos", r"trikompart\w*",
              r"pangonarthr\w*", r"pangonartr\w*", r"panartros\w*", r"все? три", r"τρι\w*διαμερισ\w*",
              r"uc kompart\w*", r"trikompartman\w*", r"tous les compartiments",
              r"drie compartiment\w*")

EFFUSION = rx(r"effusion\w*", r"derrame\w*", r"epanchement\w*", r"\w*ergu(?:ss|ß)\w*",
              r"hydrops", r"\w*h(?:a|e|ae)marthr\w*", r"gewrichtsvocht",
              r"vochtuitstorting", r"izljev\w*", r"излив\w*", r"συλλογ\w* υγρου", r"υδραρθρ\w*",
              r"efuzyon\w*", r"sivi artis\w*", r"sivi miktari\w*(?: \w+){0,3} art\w*", r"sıvı artış\w*", r"fluid (?:collection|increase)",
              r"joint fluid", r"opzetting", r"distension of the suprapatellar")
EFF_TRACE = rx(r"trace\b", r"physiolog\w*", r"fisiolog\w*", r"physiologisch\w*", r"fysiolog\w*",
               r"minimal\w*", r"minim[oa]\b", r"minimaal", r"ελαχιστ\w*", r"φυσιολογ\w*",
               r"физиолог\w*", r"fizyolojik", r"fizioloski\w*")
SYNOVITIS = rx(r"\w*synovit\w*", r"\w*sinovit\w*", r"\w*synovialit\w*", r"синовит\w*",
               r"υμενιτιδ\w*", r"snovit\w*", r"synovial (?:thickening|hypertroph\w*|proliferation)",
               r"engrosamiento sinovial", r"sinovial engrosad\w*", r"epaississement synovial",
               r"synoviale verdick\w*", r"synoviale verdikking", r"sinovi\w* zadeb\w*",
               r"sinoviumda kalinlas\w*", r"sinoviyal kalinlas\w*", r"synovial\w* hypertrophie",
               r"(?:thicken\w*|verdickt\w*|engrosa\w*|zadebljan\w*|epaissi\w*|verdikt\w*|"
               r"hypertroph\w*|παχυνσ\w*)(?: \w+){0,2} (?:of the )?(?:synovi|sinovi|υμεν)\w*")
BAKER = rx(r"[bβ]aker\w*", r"\w*bakerzyste\w*", r"bekerov\w*", r"бекеров\w*", r"бейкър\w*",
           r"popliteal cyst\w*", r"quistes? popliteo\w*", r"kystes? poplite\w*",
           r"poplitea\w* cist\w*", r"poplitealn\w* cist\w*", r"поплитеалн\w* кист\w*",
           r"popliteal\w*zyste\w*", r"popliteale cyste", r"popliteal kist\w*",
           r"ιγνυακ\w* κυστ\w*", r"semimembran\w*[- ]gastro\w* bursa\w*",
           r"gastro\w*[- ]semimembran\w* bursa\w*")
CONTUSION = rx(r"contusi\w*", r"bone bruis\w*", r"bruis\w*", r"kontusion\w*", r"\w*kontusion\w*",
               r"kneuzing\w*", r"botkneuzing\w*", r"kontuzij\w*", r"kontuzi\w*", r"контузи\w*",
               r"θλασ\w* οστ\w*", r"οστικ\w* θλασ\w*", r"kontuzyon\w*", r"ezilme\w*")
BME = rx(r"bone marrow (?:o?edema|oedema|signal)", r"marrow o?edema", r"bone o?edema",
         r"edema (?:oseo|de (?:la )?medula osea)", r"edema oseo", r"oedeme osseux",
         r"oedeme (?:de la )?(?:moelle|medullaire)", r"\w*knochenmark\w*odem\w*", r"knochenodem\w*",
         r"botoedeem", r"beenmergoedeem", r"kostan\w* edem\w*", r"edem\w* kost\w*",
         r"костно-?\s?мозъч\w* едем", r"костномозъч\w* едем", r"οστεομυελικ\w* οιδημ\w*",
         r"οιδημα (?:του )?μυελου", r"kemik iligi odem\w*", r"kemik iliği ödem\w*")
TRAUMA = rx(r"\w*traum\w*", r"impact\w*", r"kissing", r"pivot", r"травм\w*", r"τραυμα\w*",
            r"travm\w*", r"posttraum\w*", r"postraum\w*", r"contrecoup")
FRACTURE = rx(r"\w*fractur\w*", r"\w*fractuur\w*", r"\w*fraktur\w*", r"\w*fraktür\w*", r"\w*frakture?\w*",
              r"фрактур\w*", r"счупван\w*", r"καταγμ\w*", r"kirik\w*", r"kırık\w*",
              r"infraktion\w*", r"breuk\w*", r"prijelom\w*", r"knochern\w* ausriss\w*",
              r"bony avulsion", r"avulsion fracture", r"arrachement osseux", r"avulsion osseuse",
              r"segond\b", r"impressionsfraktur", r"insufficiency fracture")

SENT_SPLIT = re.compile(r"(?<!\d)\.(?!\d)|\n|;|•|^\s*-\s", re.M)
CLAUSE_SPLIT = re.compile(r",|:|\b(?:but|however|whereas|pero|sin embargo|aunque|aber|jedoch|"
                          r"maar|echter|mais|cependant|ali|међутим|но|αλλα|ομως|ancak|fakat|ama)\b")
WORD = re.compile(r"\w+|\?")


# ---------------------------------------------------------------- utilidades de texto

def strip_clinical(text):
    """Quita los bloques de indicación/antecedentes hasta el siguiente encabezado."""
    out, pos = [], 0
    for m in CLIN_HDR.finditer(text):
        if m.start() < pos:
            continue
        out.append(text[pos:m.start()])
        nxt = BODY_HDR.search(text, m.end())
        if nxt is None:
            nl = text.find("\n", m.end())
            pos = len(text) if nl == -1 else nl
        else:
            pos = nxt.start()
    out.append(text[pos:])
    return " ".join(out)


def split_sections(text):
    m = CONCL_HDR.search(text)
    if m is None:
        return text, ""
    return text[:m.start()], text[m.end():]


def clauses_of(sentence):
    return [c for c in CLAUSE_SPLIT.split(sentence) if c and c.strip()]


def n_words(s):
    return len(WORD.findall(s))


def mention_status(clause, start, end):
    """Estado de un término encontrado en clause[start:end]: neg | unc | pos."""
    for m in NEG_PRE.finditer(clause, 0, start):
        if n_words(clause[m.end():start]) <= 7:
            return "neg"
    for m in NEG_POST.finditer(clause, end):
        if n_words(clause[end:m.start()]) <= 5:
            return "neg"
    if UNCERTAIN.search(clause):
        return "unc"
    return "pos"


def side_of(text_window):
    sides = set()
    if MED_W.search(text_window):
        sides.add("medial")
    if LAT_W.search(text_window):
        sides.add("lateral")
    return sides


def nearby(clause, start, end, n_before=5, n_after=6):
    """Texto de las n palabras antes y después de un span."""
    before = WORD.findall(clause[:start])[-n_before:]
    after = WORD.findall(clause[end:])[:n_after]
    return " ".join(before), " ".join(after)


# ---------------------------------------------------------------- reglas por etiqueta
# Cada regla recibe (sentence_clauses, i) y devuelve [(label, status, weak, evidence)]

def finding_with_anatomy(label, anat, finding, extra=None):
    def rule(cl, i):
        out = []
        here = cl[i]
        has_anat = anat.search(here) or (i > 0 and anat.search(cl[i - 1]))
        if not has_anat:
            return out
        for pat in [finding] + ([extra] if extra is not None else []):
            for m in pat.finditer(here):
                out.append((label, mention_status(here, m.start(), m.end()), pat is extra,
                            here.strip()))
        return out
    return rule


def meniscus_rule(cl, i):
    out = []
    here = cl[i]
    for m in TEAR.finditer(here):
        status = mention_status(here, m.start(), m.end())
        # cláusula con el menisco: la propia, la anterior o (si la rotura no habla
        # de otra estructura) una posterior: "rotura del cuerno anterior, ..., del menisco lateral"
        cands = [here] + ([cl[i - 1]] if i > 0 else [])
        if not MENISCUS.search(here) and not OTHER_STRUCT.search(here):
            cands += cl[i + 1:]
        cands = [c for c in cands if MENISCUS.search(c)]
        if not cands:
            continue
        sides = set()
        if BOTH_MENISCI.search(cands[0]):
            sides = {"medial", "lateral"}
        else:
            # lateralidad cerca de la palabra menisco
            for src in cands:
                for mm in MENISCUS.finditer(src):
                    b, a = nearby(src, mm.start(), mm.end(), 4, 4)
                    sides |= side_of(b + " " + a + " " + mm.group(0))
                if sides:
                    break
            if not sides:  # último recurso: toda la oración
                sides = side_of(" ".join(cl))
                if len(sides) > 1:
                    sides = set()
        for s in sides:
            out.append(("Medial Meniscus" if s == "medial" else "Lateral Meniscus",
                        status, False, here.strip()))
    return out


def oa_rule(cl, i):
    out = []
    sentence = " ".join(cl)
    here = cl[i]
    for pat, weak in ((OA, False), (OA_WEAK, True)):
        if weak and OA_EXCLUDE_CTX.search(here):
            continue
        for m in pat.finditer(here):
            status = mention_status(here, m.start(), m.end())
            comps = set()
            if COMP_ALL.search(sentence):
                comps = {"Medial OA", "Lateral OA", "PF OA"}
            if COMP_PF.search(sentence) or "retropatell" in m.group(0) \
                    or "femoropatell" in m.group(0):
                comps.add("PF OA")
            # medial/lateral cuentan como tibiofemoral si no van pegados a rótula/faceta
            for side_pat, lab in ((MED_W, "Medial OA"), (LAT_W, "Lateral OA")):
                for sm in side_pat.finditer(sentence):
                    b, a = nearby(sentence, sm.start(), sm.end(), 3, 3)
                    if not PATELLA_W.search(b + " " + a):
                        comps.add(lab)
                        break
            if not comps and not weak and "gonartro" in m.group(0):
                comps = {"Medial OA"} if MED_W.search(sentence) else set()
            for c in comps:
                out.append((c, status, weak, here.strip()))
    return out


def simple_rule(label, pat, weak_pat=None):
    def rule(cl, i):
        out = []
        here = cl[i]
        for m in pat.finditer(here):
            out.append((label, mention_status(here, m.start(), m.end()), False, here.strip()))
        if weak_pat is not None:
            for m in weak_pat.finditer(here):
                out.append((label, mention_status(here, m.start(), m.end()), True, here.strip()))
        return out
    return rule


def effusion_rule(cl, i):
    out = simple_rule("Effusion", EFFUSION)(cl, i)
    if out and EFF_TRACE.search(cl[i]):
        # derrame mínimo/fisiológico: no es "exceso de líquido" -> incierto
        out = [(l, "neg" if s == "neg" else "unc", w, e) for l, s, w, e in out]
    return out


def contusion_rule(cl, i):
    out = simple_rule("Contusion", CONTUSION)(cl, i)
    sentence = " ".join(cl)
    here = cl[i]
    for m in BME.finditer(here):
        status = mention_status(here, m.start(), m.end())
        if TRAUMA.search(sentence):
            out.append(("Contusion", status, True, here.strip()))
        elif status == "pos":
            out.append(("Contusion", "unc", True, here.strip()))
    return out


RULES = [
    finding_with_anatomy("ACL", ACL, TEAR, SPRAIN),
    finding_with_anatomy("MCL", MCL, TEAR, SPRAIN),
    finding_with_anatomy("MCL", MCL, EDEMA),
    meniscus_rule,
    oa_rule,
    effusion_rule,
    simple_rule("Synovitis", SYNOVITIS),
    simple_rule("Baker's", BAKER),
    contusion_rule,
    simple_rule("Fracture", FRACTURE),
]


# ---------------------------------------------------------------- extracción y agregación

def mentions(section_text, section_name):
    out = []
    for sent in SENT_SPLIT.split(section_text):
        if not sent or not sent.strip():
            continue
        cl = clauses_of(sent)
        for i in range(len(cl)):
            for rule in RULES:
                for label, status, weak, ev in rule(cl, i):
                    out.append((label, status, weak, section_name, ev))
    return out


def aggregate(ms, has_concl, short):
    """Combina las menciones de una etiqueta en (label, conf, status)."""
    by_sec = {"concl": {"pos": [], "neg": [], "unc": []}, "body": {"pos": [], "neg": [], "unc": []}}
    for status, weak, sec in ms:
        by_sec[sec][status].append(weak)
    concl, body = by_sec["concl"], by_sec["body"]
    strong = lambda xs: any(not w for w in xs)  # noqa: E731

    if strong(concl["pos"]):
        return 1, 0.95, "pos_concl"
    if strong(body["pos"]):
        if has_concl and concl["neg"]:
            return 0, 0.60, "conflict"
        return 1, 0.85, "pos_body"
    if concl["pos"] or body["pos"]:
        return 1, 0.70, "pos_weak"
    if concl["unc"] or body["unc"]:
        return 1, 0.50, "uncertain"
    if concl["neg"] or body["neg"]:
        return 0, 0.95, "neg"
    return 0, (0.60 if short else 0.80), "absent"


def label_report(report):
    text = strip_clinical(normalize(report))
    body, concl = split_sections(text)
    ms = mentions(body, "body") + mentions(concl, "concl")
    has_concl, short = bool(concl.strip()), len(text) < 300
    row, ev = {}, {}
    for L in LABELS:
        mine = [(s, w, sec) for (lab, s, w, sec, e) in ms if lab == L]
        y, conf, status = aggregate(mine, has_concl, short)
        row[L], row[f"{L}_conf"], row[f"{L}_status"] = y, conf, status
        row[f"{L}_p"] = conf if y == 1 else 1 - conf
        ev[L] = " || ".join(sorted({f"[{sec}:{s}] {e[:160]}"
                                    for (lab, s, w, sec, e) in ms if lab == L}))[:600]
    return row, ev


def main():
    train = pd.read_csv(META / "train.csv")
    rows, evs = [], []
    for uid, rep in zip(train.StudyInstanceUID, train.Report):
        row, ev = label_report(rep)
        rows.append({"StudyInstanceUID": uid, **row})
        evs.append({"StudyInstanceUID": uid, **ev})
    out = pd.DataFrame(rows)
    out.to_csv(META / "weak_labels.csv", index=False)
    pd.DataFrame(evs).to_csv(META / "weak_labels_evidence.csv", index=False)

    print("Prevalencia débil (todos los estudios, inciertas = 1):")
    print(out[LABELS].mean().round(3).to_string())
    print("\nEstado por etiqueta:")
    print(pd.concat({L: out[f"{L}_status"].value_counts() for L in LABELS}, axis=1)
          .fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
