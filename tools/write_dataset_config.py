from collections import Counter
import contextlib
import os
import json

def _yaml_scalar(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def _yaml_inline(value):
    if isinstance(value, list):
        return '[' + ', '.join(_yaml_inline(v) for v in value) + ']'
    return _yaml_scalar(value)


def _write_simple_yaml(path, data):
    lines = []

    def emit(key, value, indent=0):
        pad = ' ' * indent
        if isinstance(value, dict):
            lines.append(f'{pad}{key}:')
            for child_key, child_value in value.items():
                emit(child_key, child_value, indent + 2)
        else:
            lines.append(f'{pad}{key}: {_yaml_inline(value)}')

    for key, value in data.items():
        emit(key, value)

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_dataset_config(metadata_path,
                         output_dir,
                         snr_values,
                         simulation,
                         noise,
                         sample_rate,
                         room_sizes,
                         rt60_values,
                         num_channels,
                         target_radius,
                         target_azimuth_deg,
                         interferer_radius,
                         interferer_azimuth_deg):
    metadata = []
    if os.path.isfile(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = [json.loads(line) for line in f if line.strip()]

    rt60_seen = [entry['rt60'] for entry in metadata if entry.get('rt60') is not None]
    snr_seen = sorted({entry.get('snr') for entry in metadata if entry.get('snr') is not None})
    num_interferers = [len(entry.get('interference_positions', [])) for entry in metadata]

    dataset_folder = os.path.dirname(output_dir)
    config_path = os.path.join(dataset_folder, 'dataset_config.yaml')
    dataset_name = os.path.basename(dataset_folder)

    config = {
        'dataset_name': dataset_name,
        'metadata_file': os.path.basename(metadata_path),
        'config_note': 'Created automatically by dataset_generator.py after generation. Radius/azimuth entries describe the generation expressions used by this script.',
        'snr_values': snr_values,
        'simulation': simulation,
        'noise': noise,
        'sample_rate': sample_rate,
        'room_sizes': room_sizes,
        'rt60_values_requested': rt60_values,
        'num_channels': num_channels,
        'snr_values_seen_in_metadata': snr_seen,
        'num_files_generated': len(metadata),
        'actual_rt60_range': {
            'min': round(min(rt60_seen), 3) if rt60_seen else None,
            'max': round(max(rt60_seen), 3) if rt60_seen else None,
        },
        'num_interferers': {
            'options': sorted(set(num_interferers)),
            'counts': dict(sorted(Counter(num_interferers).items())),
        },
        'target_radius': target_radius,
        'target_azimuth_deg': target_azimuth_deg,
        'interferer_radius': interferer_radius,
        'interferer_azimuth_deg': interferer_azimuth_deg,
    }

    _write_simple_yaml(config_path, config)
    print(f'Saved dataset configuration to {config_path}')
