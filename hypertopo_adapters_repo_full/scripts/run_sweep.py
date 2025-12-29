import yaml, os, copy, subprocess, sys

def run(cfg_path, overrides, log_dir):
    with open(cfg_path, 'r') as f:
        base = yaml.safe_load(f)
    cfg = copy.deepcopy(base)
    for k,v in overrides.items():
        keys = k.split('.'); d = cfg
        for kk in keys[:-1]: d = d[kk]
        d[keys[-1]] = v
    cfg['logging']['out_dir'] = os.path.join(log_dir, 'run_' + '_'.join(f"{k.replace('.','-')}-{v}" for k,v in overrides.items()))
    os.makedirs(cfg['logging']['out_dir'], exist_ok=True)
    tmp_cfg = os.path.join(cfg['logging']['out_dir'], 'config.yaml')
    with open(tmp_cfg, 'w') as f: yaml.safe_dump(cfg, f)
    print('Launching', tmp_cfg)
    subprocess.run([sys.executable, '-m', 'src.train', '--config', tmp_cfg], check=False)

def main():
    with open('configs/default.yaml', 'r') as f: base = yaml.safe_load(f)
    with open('configs/sweep.yaml', 'r') as f: sweep = yaml.safe_load(f)
    log_dir = os.path.join(base['logging']['out_dir'], 'sweeps'); os.makedirs(log_dir, exist_ok=True)
    grid = []
    for tw in sweep['sweep']['topology_weight']:
        for dh in sweep['sweep']['dh']:
            for ds in sweep['sweep']['ds']:
                for temp in sweep['sweep']['temperature']:
                    grid.append({'loss.topology_weight': tw, 'model.latent_dims.dh': dh, 'model.latent_dims.ds': ds, 'loss.contrastive.temperature': temp})
    for run_overrides in grid: run('configs/default.yaml', run_overrides, log_dir)

if __name__ == '__main__':
    main()
