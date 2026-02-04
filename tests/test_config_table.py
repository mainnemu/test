from vdp_protocol.config_table import CONFIGS, ID_TO_INDEX, PAIR_IDS

def test_config_integrity():
    # 12 configs
    assert len(CONFIGS) == 12
    # IDs unique
    ids = set(c.id for c in CONFIGS)
    assert len(ids) == 12
    # Recip mapping
    for c in CONFIGS:
        recip = CONFIGS[ID_TO_INDEX[c.recip_id]]
        assert recip.recip_id == c.id

def test_pair_definitions():
    # 6 pairs
    assert len(PAIR_IDS) == 6
    # Check if they match the config table reciprocity
    for id_a, id_b in PAIR_IDS:
        idx_a = ID_TO_INDEX[id_a]
        assert CONFIGS[idx_a].recip_id == id_b
