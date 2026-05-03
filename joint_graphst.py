"""
joint_graphst.py

Novel semi-supervised extension of GraphST for spatial transcriptomics.

Extends GraphST (Long et al., Nature Communications 2023) by adding a 
supervised classification loss to the self-supervised contrastive pretraining
objective, enabling cortical layer prediction from spatial gene expression data.

Original GraphST loss (Long et al., 2023):
    L = λ1 * L_recon + λ2 * (L_SCL + L_SCL_corrupt)

Our novel semi-supervised extension:
    L = λ1 * L_recon + λ2 * (L_SCL + L_SCL_corrupt) + γ * L_cls

Where:
    L_recon       = MSE reconstruction loss (unsupervised)
    L_SCL         = spatial contrastive loss on real graph (unsupervised)
    L_SCL_corrupt = spatial contrastive loss on corrupted graph (unsupervised)
    L_cls         = cross-entropy classification loss on labeled spots (supervised)
    γ             = supervision weight (our novel hyperparameter)
                    γ=0: reduces to original GraphST
                    γ>0: semi-supervised extension (our contribution)

References:
    Long et al. (2023). GraphST. Nature Communications.
    Kipf & Welling (2017). Semi-supervised GCN. ICLR.
    You et al. (2020). Graph contrastive learning. NeurIPS.
    Rong et al. (2020). Self-supervised graph transformer. NeurIPS.

Author: Yossi Moff, Yale University CPSC 4830
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
from torch.optim import Adam
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GraphST'))
from GraphST.preprocess import (preprocess_adj, preprocess, construct_interaction,
                                 add_contrastive_label, get_feature, permutation, fix_seed)
from GraphST.model import Encoder


# ── MLP Classifier ────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Two-layer MLP classification head applied to GraphST embeddings.
    
    Used in Stage 2 of our architecture to predict cortical layer identity
    from the 64-dimensional spatial embeddings produced by the GraphST encoder.
    
    Includes BatchNorm and Dropout for regularization, and supports
    class-weighted cross-entropy to handle cortical layer imbalance
    (e.g. L3 has 3x more spots than L2 in the DLPFC dataset).
    """
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

# ── Train MLP ────────────────────────────────────────────────────────────────

def train_mlp(X_train, y_train, X_val, y_val, input_dim, num_classes,
              hidden_dim=256, epochs=300, lr=1e-3):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class weights to handle imbalance (cortical layers have very different sizes)
    class_counts = np.bincount(y_train)
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_v  = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_f1, best_state = 0, None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_v).argmax(dim=1).cpu().numpy()
                val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val Macro-F1: {val_f1:.4f}")

    model.load_state_dict(best_state)
    return model


# ── Joint GraphST Model ───────────────────────────────────────────────────────

class JointGraphST(nn.Module):
    """
    Novel semi-supervised extension of GraphST.
    
    Combined loss (our contribution):
        L = alpha * recon_loss + beta * contrastive_loss + gamma * classification_loss
    
    When gamma=0: reduces to original GraphST (purely unsupervised)
    When gamma>0: supervision signal guides spatial embeddings toward
                  cortical layer discriminability
                  
    Reference: Kipf & Welling (ICLR 2017) semi-supervised GCN framework
    """
    def __init__(self, dim_input, dim_output, graph_neigh, num_classes, 
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(dim_input, dim_output, graph_neigh)
        self.classifier = nn.Sequential(
            nn.Linear(dim_output, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features, features_a, adj):
        hidden_feat, emb, ret, ret_a = self.encoder(features, features_a, adj)
        # Use hidden_feat (64-dim latent) not emb (reconstructed input)
        logits = self.classifier(hidden_feat)
        return hidden_feat, emb, ret, ret_a, logits


# ── Training Function ─────────────────────────────────────────────────────────

def run_joint_graphst(adata_input, labels, train_mask, test_mask, 
                       n_classes, gamma, device, 
                       epochs=600, alpha=10, beta=1, random_seed=41):
    """
    Train joint GraphST with given gamma and return test metrics.
    gamma=0 → pure unsupervised (original GraphST)
    gamma>0 → semi-supervised (our contribution)
    """
    fix_seed(random_seed)
    
    # Fresh copy of adata for each run
    adata_run = adata_input.copy()
    
    # Run GraphST preprocessing
    if 'highly_variable' not in adata_run.var.keys():
        preprocess(adata_run)
    if 'adj' not in adata_run.obsm.keys():
        construct_interaction(adata_run)
    if 'label_CSL' not in adata_run.obsm.keys():
        add_contrastive_label(adata_run)
    if 'feat' not in adata_run.obsm.keys():
        get_feature(adata_run)

    features = torch.FloatTensor(adata_run.obsm['feat'].copy()).to(device)
    features_a = torch.FloatTensor(adata_run.obsm['feat_a'].copy()).to(device)
    label_CSL = torch.FloatTensor(adata_run.obsm['label_CSL']).to(device)
    adj_mat = adata_run.obsm['adj']
    graph_neigh = torch.FloatTensor(
        adata_run.obsm['graph_neigh'].copy() + np.eye(adj_mat.shape[0])
    ).to(device)
    adj_mat = preprocess_adj(adj_mat)
    adj_tensor = torch.FloatTensor(adj_mat).to(device)

    dim_input = features.shape[1]
    dim_output = 64

    # Class weights for imbalance
    class_counts = np.bincount(labels[train_mask])
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)

    # Initialize joint model
    model = JointGraphST(dim_input, dim_output, graph_neigh, n_classes).to(device)
    
    loss_CSL = nn.BCEWithLogitsLoss()
    loss_CE = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

    labels_tensor = torch.LongTensor(labels).to(device)
    train_mask_tensor = torch.BoolTensor(train_mask).to(device)

    # Training loop
    for epoch in tqdm(range(epochs), desc=f"gamma={gamma}"):
        model.train()
        features_a = permutation(features)
        hidden, emb, ret, ret_a, logits = model(features, features_a, adj_tensor)

        # Original GraphST losses
        loss_sl_1 = loss_CSL(ret, label_CSL)
        loss_sl_2 = loss_CSL(ret_a, label_CSL)
        loss_feat = F.mse_loss(features, emb)

        # Novel classification loss (only on labeled spots)
        loss_cls = loss_CE(logits[train_mask_tensor], 
                           labels_tensor[train_mask_tensor])

        # Combined loss — key novel contribution
        # gamma=0 → original GraphST, gamma>0 → semi-supervised
        loss = alpha * loss_feat + beta * (loss_sl_1 + loss_sl_2) + gamma * loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        features_a_eval = permutation(features)
        _, emb_final, _, _, logits_final = model(features, features_a_eval, adj_tensor)
        
        test_logits = logits_final[test_mask]
        test_probs = F.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        test_true = labels[test_mask]

        # Clustering metrics on full embeddings
        emb_np = np.nan_to_num(emb_final.cpu().numpy())
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(emb_np)
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)

        # Classification metrics
        macro_f1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
        y_bin = label_binarize(test_true, classes=range(n_classes))
        auroc = roc_auc_score(y_bin, test_probs, multi_class='ovr', average='macro')

    return {
        'gamma': gamma,
        'macro_f1': macro_f1,
        'auroc': auroc,
        'ari': ari,
        'nmi': nmi
    }

# ── Evaluation Function ───────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, n_classes, device, model_name="Model"):
    """
    Evaluate a trained model on test data.
    
    Computes Macro-F1 and AUROC, which are the primary metrics
    for this project due to cortical layer class imbalance.
    Macro-F1 weights all classes equally regardless of size.
    AUROC captures minority-class performance.
    
    Parameters
    ----------
    model : JointGraphST or MLPClassifier
        Trained model
    X_test : array
        Test embeddings
    y_test : array
        True labels
    n_classes : int
        Number of classes
    device : torch.device
        CPU or CUDA
    model_name : str
        Name for printing
        
    Returns
    -------
    macro_f1 : float
    auroc : float
    """
    model.eval()
    X_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        # Handle both JointGraphST and standalone MLPClassifier
        if hasattr(model, 'classifier'):
            logits = model.classifier(X_t)
        else:
            logits = model(X_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    y_bin = label_binarize(y_test, classes=range(n_classes))
    auroc = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')

    print(f"\n=== {model_name} ===")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"AUROC    : {auroc:.4f}")
    
    return macro_f1, auroc
