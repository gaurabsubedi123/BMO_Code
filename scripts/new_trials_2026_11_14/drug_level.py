"""Which registry lines describe drug-level measures: concentrations or kinetics of the study drug
(pharmacokinetics) and antibodies against the study drug (anti-drug antibodies).

Built from analyze.py's DRUG pattern, with a wider guard so that measurements of the body's own analytes
(e.g. 'plasma concentration of GFAP', albumin, immunoglobulins), MRI and EEG measures, PET receptor occupancy and
antibody responses to an anti-amyloid or anti-tau vaccine are not counted as drug-level. Decided from wording only,
never from labels; the guard terms were fixed after reading every set-aside line that annotators had marked as a biomarker.
"""
import pathlib, re
import pandas as pd

_src = (pathlib.Path(__file__).with_name("analyze.py")).read_text()
ns = {}
exec(_src.split("for name in")[0], ns)
DRUG = ns["DRUG"]
DRUG2 = re.compile(DRUG.pattern + r"|concentration of [a-z0-9-]+mab|antibodies in serum|anti-?[a-z0-9-]+ antibodies|"
                   r"(csf|serum|blood|plasma) [a-z0-9-]+ concentration|\bcmin\b|\bcav", re.I)
GUARD = re.compile(
    r"amyloid|a[βb]\s?\d|abeta|anti-a[βb]|\btau\b|p-?tau|t-?tau|mtbr|nfl|neurofilament|apoe|sapp|biomarker|gfap|ykl|trem2|"
    r"alpha-?synuclein|tnf|tumou?r necrosis|interleukin|\bil-?\d|cytokine|mcp-?1|chemoattractant|c-reactive|\bcrp\b|"
    r"inflammat|oxidative|dopamine|glutamate|receptor occupancy|occupancy|\beeg\b|event related|p300|evoked potential|"
    r"glial fibrillary|albumin|\bmri\b|magnetic resonance|axonal|immunoglobulin|multitep|vaccin", re.I)


def drug(texts):
    s = pd.Series(texts).fillna("")
    return (s.str.contains(DRUG2) & ~s.str.contains(GUARD)).values
