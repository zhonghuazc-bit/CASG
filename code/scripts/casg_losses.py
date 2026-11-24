



import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'loss_configs'
DEFAULT_MAIN_LOSS = 'equal'


def _safe_r(m):
    
    r = m.get('pearson_r')
    return r if r == r else 0.5


def build_loss(terms):
    
    def loss(m):
        total = 0.0
        for t in terms:
            w = float(t['weight'])
            metric = t['metric']
            kind = t.get('kind', 'abs_diff')
            if kind == 'abs_diff':
                target = float(t['target'])
                dev = abs(m[metric] - target)
                if t.get('deviation', 'absolute') == 'relative':
                    dev = dev / target
                total += w * dev
            elif kind == 'one_minus':
                total += w * (1 - _safe_r(m))
            else:
                raise ValueError(f'未知损失项 kind: {kind} (配置 metric={metric})')
        return total
    return loss


def load_loss_configs(config_dir=None):
    
    cfg_dir = Path(config_dir) if config_dir else CONFIG_DIR
    cfgs = {}
    for fp in sorted(cfg_dir.glob('*.json')):
        cfg = json.load(open(fp, encoding='utf-8'))
        name = cfg.get('name') or fp.stem
        if not cfg.get('terms'):
            continue
        if name in cfgs:
            print(f'[警告] 损失名 {name!r} 重复: {fp} 跳过, 保留 {cfgs[name]["source"]}')
            continue
        cfgs[name] = dict(func=build_loss(cfg['terms']),
                          label=cfg.get('label', name),
                          config=cfg,
                          source=str(fp))
    return cfgs


def resolve_name(s):
    
    s = s.strip()
    if s in LOSS_FUNCTIONS:
        return s
    p = Path(s)
    if p.suffix.lower() == '.json' and p.exists():
        target = p.resolve()
        for name, meta in LOSS_FUNCTIONS.items():
            if Path(meta['source']).resolve() == target:
                return name
    raise ValueError(f'无法解析损失配置: {s}; 可用: {sorted(LOSS_FUNCTIONS.keys())}')



LOSS_FUNCTIONS = load_loss_configs()


def compute_loss_values(m):
    
    return {name: float(meta['func'](m)) for name, meta in LOSS_FUNCTIONS.items()}


if __name__ == '__main__':
    print('已加载损失配置:')
    for name, meta in LOSS_FUNCTIONS.items():
        print(f'  {name:<12} <- {meta["source"]}')
        print(f'               {meta["label"]}')
