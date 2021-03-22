def get_save_filename(prefix, cfg, it=0):
    difficulty = '_' + cfg.env.settings.difficulty
    noise = '_noise' if cfg.env.observation.with_noise else ''
    tactile = '_tactile' if cfg.env.observation.with_tactile_sensor else ''
    force = '_force' if cfg.env.observation.with_force else ''
    rs = "_rs_" + str(it) if hasattr(cfg.train, 'num_random_seeds') and cfg.train.num_random_seeds > 1 else ''
    save_filename = prefix + difficulty + noise + tactile + force + rs
    return save_filename