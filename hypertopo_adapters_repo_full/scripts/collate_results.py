import argparse, os, glob, json, yaml, csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/sweeps')
    ap.add_argument('--out_csv', default='outputs/sweeps/summary.csv')
    ap.add_argument('--latex', default='outputs/sweeps/table.tex')
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, 'run_*'))):
        cfgp = os.path.join(d, 'config.yaml')
        metp = os.path.join(d, 'best_metrics.json')
        if not (os.path.exists(cfgp) and os.path.exists(metp)):
            continue
        with open(cfgp, 'r') as f: cfg = yaml.safe_load(f)
        with open(metp, 'r') as f: met = json.load(f)
        row = {
            'dir': d,
            'topology_weight': cfg['loss']['topology_weight'],
            'dh': cfg['model']['latent_dims']['dh'],
            'ds': cfg['model']['latent_dims']['ds'],
            'temp': cfg['loss']['contrastive']['temperature'],
            'dice': met.get('dice'),
            'iou': met.get('iou'),
            'boundary_f1': met.get('boundary_f1'),
            'betti0_err': met.get('betti0_err'),
            'betti1_err': met.get('betti1_err'),
            'pd_dist': met.get('pd_dist'),
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows: w.writeheader(); w.writerows(rows)
    print('Wrote CSV:', args.out_csv)

    # Minimal LaTeX table
    
    #with open(args.latex, 'w') as f:
    #    f.write('\begin{tabular}{lcccccc}\n')
    #    f.write('\toprule\n')
    #    f.write('Setting & Dice $\uparrow$ & IoU $\uparrow$ & B-F1 $\uparrow$ & $\Delta\beta_0$ $\downarrow$ & $\Delta\beta_1$ $\downarrow$ & PD $\downarrow$ \\\n')
    #    f.write('\midrule\n')
    #    for r in rows:
    #        setting = f"tw={r['topology_weight']}, dh={r['dh']}, ds={r['ds']}, T={r['temp']}"
    #        f.write(f"{setting} & {r['dice']:.3f} & {r['iou']:.3f} & {r['boundary_f1']:.3f} & {r['betti0_err']:.2f} & {r['betti1_err']:.2f} & {r['pd_dist'] if r['pd_dist'] is not None else 'NaN'} \\\n")
    #    f.write('\bottomrule\n')
    #    f.write('\end{tabular}\n')
    #print('Wrote LaTeX table:', args.latex)
    
    # --- replace the LaTeX-writing block in scripts/collate_results.py with this ---
    
    with open(args.latex, 'w') as f:
       f.write(r'\begin{tabular}{lcccccc}' + '\n')
       f.write(r'\toprule' + '\n')
       header = r'Setting & Dice $\uparrow$ & IoU $\uparrow$ & B-F1 $\uparrow$ & $\Delta\beta_0$ $\downarrow$ & $\Delta\beta_1$ $\downarrow$ & PD $\downarrow$ \\'
       f.write(header + '\n')
       f.write(r'\midrule' + '\n')
       for r in rows:
          setting = f"tw={r['topology_weight']}, dh={r['dh']}, ds={r['ds']}, T={r['temp']}"
          line = f"{setting} & {r['dice']:.3f} & {r['iou']:.3f} & {r['boundary_f1']:.3f} & {r['betti0_err']:.2f} & {r['betti1_err']:.2f} & {r['pd_dist'] if r['pd_dist'] is not None else 'NaN'} \\\\"
          f.write(line + '\n')
       f.write(r'\bottomrule' + '\n')
       f.write(r'\end{tabular}' + '\n')
    

if __name__ == '__main__':
    main()
