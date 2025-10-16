import numpy as np
from src.train.config import CFG
from src.common.db import return_scalar, return_all, latest_snapshot_id

# Return a split configuration id by name or latest for a given snapshot 
def get_split_configuration_id(snapshot_id, name=None):
    if name: 
        sql = """
            SELECT id 
            FROM split_configurations
            WHERE snapshot_id=%s AND name=%s
            ORDER BY created_at DESC
            LIMIT 1
        """

        row = return_scalar(sql, [snapshot_id, name])
        if not row: raise RuntimeError(f'No split_configurations found for snapshot_id={snapshot_id}, name={name}. Run transform')
    else:
        sql = """
            SELECT id
            FROM split_configurations
            WHERE snapshot_id=%s
            ORDER BY created_at DESC
            LIMIT 1
        """
        row = return_scalar(sql, [snapshot_id])
        if not row: raise RuntimeError(f'No split_configurations found for snapshot_id={snapshot_id}. Run transform')
    
    return int(row)

# Returns int64 numpy arrays of train iamge ids and val image ids for a given split configuration
def get_train_val_image_ids(split_configuration_id):
    sql = """
        SELECT image_id, split
        FROM train_val_splits
        WHERE split_configuration_id=%s
    """

    rows = return_all(sql, [split_configuration_id])
    if not rows: 
        raise RuntimeError(f'No train_val_splits rows for split_configuration_id={split_configuration_id}. Check transform step')

    train_ids = np.array([r[0] for r in rows if r[1] == 'train'])
    val_ids = np.array([r[0] for r in rows if r[1] == 'val'])
    if train_ids.size == 0 or val_ids.size == 0:
        raise RuntimeError('Empty or Non-existant train or val split. Check transform step')
    
    return np.array(train_ids, dtype=np.int64), np.array(val_ids, dtype=np.int64) 

# Get raw pixels (BYTEA) and labels for the given Image ids for the snapshot
# Return 2 np arrays X and y for training pixels and labels respectively
def get_pixels_and_labels_by_ids(snapshot_id, image_ids):
    sql = """
        SELECT id, label, pixels
        FROM mnist_images_train
        WHERE snapshot_id=%s AND id = ANY(%s)
        ORDER BY id
    """
    rows = return_all(sql, [snapshot_id, list(map(int, image_ids.tolist()))])
    if not rows:
        raise RuntimeError('No matching mnist_images_train rows for given images ids')
    
    N = len(rows)
    X = np.empty((N, 28, 28, 1), dtype=np.uint8)
    y = np.empty((N,), dtype=np.int64)

    for i, (_, label, pixels) in enumerate(rows):
        arr = np.frombuffer(pixels, dtype=np.uint8)
        if arr.size != 784:
            raise RuntimeError(f'Unexpected pixel length {arr.size}, expected 784')
        X[i, :, :, 0] = arr.reshape(28, 28)
        y[i] = int(label)
    
    return X, y

# Gets snapshot id or uses existing 
# Gets split configuration id or uses existing by name
# gets train and val images id sets
# Limits the train set by config limit or specified
# Gets pixels and labels for train and val images 
# Returns split configuration id along with train and val pixels and labels
def load_training_data(snapshot_id=None, split_name=None, limit=None):
    sid = snapshot_id if snapshot_id else latest_snapshot_id()
    scid = get_split_configuration_id(sid, split_name)

    train_ids, val_ids = get_train_val_image_ids(scid)

    if not limit: limit = CFG.limit
    if limit and limit > 0:
        train_ids = np.sort(train_ids)[: int(limit)]
    
    Xtr, ytr = get_pixels_and_labels_by_ids(sid, train_ids)
    Xva, yva = get_pixels_and_labels_by_ids(sid, val_ids)

    return scid, (Xtr, ytr), (Xva, yva)


