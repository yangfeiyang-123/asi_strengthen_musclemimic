"""Transferred training data must match a pinned, passing checkpoint contract."""
import copy
import hashlib
import json

import pytest

import musclemimic.badminton.aug100_release as release


@pytest.fixture
def transferred(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(release, 'EXPECTED_MOTION_COUNT', 2)
    manifest = tmp_path / release.DATASET_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text('train.npz\nval.npz\n')
    rows = {}
    inventory = []
    for name, split in [('train', 'train'), ('val', 'validation')]:
        path = tmp_path / (name + '.npz')
        path.write_bytes(name.encode())
        rows[name] = path
        inventory.append(dict(motion=name, split=split, cache_path=path.name,
                              cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    monkeypatch.setattr(release, '_manifest_rows', lambda: tuple(rows.items()))
    monkeypatch.setattr(release, '_validate_declared_split',
                        lambda *args, **kw: (('train',), ('val',), []))
    r = dict(schema_version=release.RELEASE_SCHEMA_VERSION, action_id=release.ACTION_ID,
             data_variant=release.CACHE_VARIANT, passed=True, errors=[],
             train_motions=['train'], validation_motions=['val'], file_inventory=inventory,
             dataset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    r['release_binding_sha256'] = release._fingerprint(r)
    q = dict(passed=True, clean_passed=True, hard_errors=[], warnings=[],
             train_motions=['train'], validation_motions=['val'],
             release_binding_sha256=r['release_binding_sha256'])
    numeric = dict(report=q, report_sha256=release._fingerprint(q))
    numeric['binding_sha256'] = release._fingerprint(numeric)
    experiment = dict(stage1_peasd_action_release_contract=r,
                      stage1_peasd_numeric_data_qc_contract=numeric)
    path = tmp_path / 'checkpoint_config.json'

    def write(value=experiment):
        path.write_text(json.dumps({'experiment': value}))
        return dict(config_path=path.name, config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    return write, experiment, rows


def test_transferred_bytes_preserve_original_contract_without_old_qc_files(transferred):
    write, experiment, _ = transferred
    r, q = release.validate_checkpoint_aug100_evidence(write(), ['train'], ['val'])
    assert r == experiment['stage1_peasd_action_release_contract']
    assert q == experiment['stage1_peasd_numeric_data_qc_contract']['report']


def test_changed_checkpoint_config_is_rejected(transferred):
    write, experiment, _ = transferred
    evidence = write()
    write({**experiment, 'changed': True})
    with pytest.raises(ValueError, match='config SHA-256'):
        release.validate_checkpoint_aug100_evidence(evidence, ['train'], ['val'])


def test_changed_trajectory_is_rejected(transferred):
    write, _, rows = transferred
    evidence = write()
    rows['val'].write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='trajectory SHA-256'):
        release.validate_checkpoint_aug100_evidence(evidence, ['train'], ['val'])


def test_swapped_split_cannot_reuse_checkpoint(transferred):
    write, experiment, _ = transferred
    experiment = copy.deepcopy(experiment)
    r = experiment['stage1_peasd_action_release_contract']
    r['train_motions'], r['validation_motions'] = ['val'], ['train']
    r['release_binding_sha256'] = release._fingerprint({k: v for k, v in r.items() if k != 'release_binding_sha256'})
    numeric = experiment['stage1_peasd_numeric_data_qc_contract']
    numeric['report']['release_binding_sha256'] = r['release_binding_sha256']
    numeric['report_sha256'] = release._fingerprint(numeric['report'])
    numeric['binding_sha256'] = release._fingerprint({k: v for k, v in numeric.items() if k != 'binding_sha256'})
    with pytest.raises(ValueError, match='split mismatch'):
        release.validate_checkpoint_aug100_evidence(write(experiment), ['train'], ['val'])


def test_failed_or_stale_numeric_qc_is_rejected(transferred):
    write, experiment, _ = transferred
    experiment['stage1_peasd_numeric_data_qc_contract']['report']['clean_passed'] = False
    with pytest.raises(ValueError, match='binding mismatch'):
        release.validate_checkpoint_aug100_evidence(write(), ['train'], ['val'])
