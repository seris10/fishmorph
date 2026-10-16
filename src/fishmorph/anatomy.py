"""Explicit anatomical definitions, separate from detection and storage.

Photo projected TL/FL are deliberately not called board-measured total/fork length.
All definitions and limitations are exported alongside measurements.
"""
LANDMARKS = {
    "snout": "Anterior snout tip, mouth closed; not the tip of an extended lower jaw.",
    "caudal_base": "Midline posterior end of the hypural/caudal-fin base; an anatomical endpoint, not a color edge.",
    "dorsal_reference": "A visible dorsal body point anterior to the caudal base, used ONLY to resolve dorsal orientation.",
    "operculum": "Most posterior bony opercular margin excluding the soft branchiostegal membrane.",
    "eye_anterior": "Anterior margin of the visible orbit in lateral view.",
    "eye_posterior": "Posterior margin of the visible orbit in lateral view.",
    "eye_center": "Center of the visible orbit, not a specular highlight.",
    "mouth_upper": "Visible upper boundary of the mouth opening at its anterior end.",
    "mouth_lower": "Visible lower boundary of the mouth opening at its anterior end.",
    "mouth_corner": "Posterior visible corner of the mouth.",
    "jaw_joint": "Jaw articulation, only when actually identifiable.",
    "dorsal_origin": "Anterior insertion of dorsal fin into the body.",
    "dorsal_insertion": "Posterior insertion of dorsal fin into the body.",
    "dorsal_apex": "Distal apex of the dorsal fin in the recorded fin pose.",
    "adipose_origin": "Anterior insertion of the adipose fin, where present.",
    "adipose_apex": "Distal apex of adipose fin, where present.",
    "pectoral_origin": "Body insertion of the visible, near-side pectoral fin.",
    "pectoral_tip": "Distal tip of the near-side pectoral fin in the recorded pose.",
    "pelvic_origin": "Body insertion of the visible, near-side pelvic fin.",
    "pelvic_tip": "Distal tip of that pelvic fin in the recorded pose.",
    "anal_origin": "Anterior body insertion of anal fin.",
    "anal_insertion": "Posterior body insertion of anal fin.",
    "anal_apex": "Distal apex of anal fin in the recorded pose.",
    "peduncle_anterior": "Midline point at the chosen anterior peduncle boundary; protocol-specific.",
    "peduncle_dorsal": "Dorsal boundary at the narrowest visible caudal-peduncle section.",
    "peduncle_ventral": "Ventral boundary at that same longitudinal section.",
    "tail_tip": "Most posterior caudal ray tip in the photographed pose; not a compressed-tail board measurement.",
    "tail_upper": "Tip of upper caudal lobe in the photographed pose.",
    "tail_lower": "Tip of lower caudal lobe in the photographed pose.",
    "tail_fork": "Posterior edge of the central caudal fork, where a fork is present.",
}

# (endpoints, denominator endpoints, label). These are Euclidean chords.
RATIOS = {
    "head_length_sl": (("snout", "operculum"), ("snout", "caudal_base"), "Head chord / SL"),
    "snout_length_head": (("snout", "eye_anterior"), ("snout", "operculum"), "Snout-to-orbit chord / head chord"),
    "eye_diameter_head": (("eye_anterior", "eye_posterior"), ("snout", "operculum"), "Visible orbit diameter / head chord"),
    "peduncle_depth_sl": (("peduncle_dorsal", "peduncle_ventral"), ("snout", "caudal_base"), "Peduncle section chord / SL"),
    "peduncle_length_sl": (("peduncle_anterior", "caudal_base"), ("snout", "caudal_base"), "Protocol-specific peduncle chord / SL"),
    "predorsal_sl": (("snout", "dorsal_origin"), ("snout", "caudal_base"), "Predorsal chord / SL"),
    "prepelvic_sl": (("snout", "pelvic_origin"), ("snout", "caudal_base"), "Prepelvic chord / SL"),
    "preanal_sl": (("snout", "anal_origin"), ("snout", "caudal_base"), "Preanal chord / SL"),
    "pectoral_length_sl": (("pectoral_origin", "pectoral_tip"), ("snout", "caudal_base"), "Visible pectoral chord / SL; pose-dependent"),
    "pelvic_length_sl": (("pelvic_origin", "pelvic_tip"), ("snout", "caudal_base"), "Visible pelvic chord / SL; pose-dependent"),
    "mouth_gape_head": (("mouth_upper", "mouth_lower"), ("snout", "operculum"), "Photographed mouth gape / head chord"),
    "caudal_span_sl": (("tail_upper", "tail_lower"), ("snout", "caudal_base"), "Caudal tip separation / SL; pose-dependent"),
}

FIN_LANDMARKS = ["dorsal_origin", "adipose_origin", "pectoral_origin", "pelvic_origin", "anal_origin"]
