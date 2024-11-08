import torch
import torch.nn as nn
import torch.nn.functional as F


class CheXpertLoss(nn.Module):
    def __init__(self, num_classes, uncertainty_strategy, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.uncertainty_strategy = uncertainty_strategy

        # Initialize loss weights as learnable parameters
        self.lambda_wbce = nn.Parameter(torch.tensor(1.0))
        self.lambda_focal = nn.Parameter(torch.tensor(1.0))
        self.lambda_asl = nn.Parameter(torch.tensor(1.0))

        # Register class weights as buffer
        if class_weights is not None:
            self.register_buffer('pos_weight', class_weights)
        else:
            self.register_buffer('pos_weight', torch.ones(num_classes))

        # Focal Loss parameters
        self.gamma = 2.0
        # Asymmetric Loss parameters
        self.gamma_pos = 1
        self.gamma_neg = 4

    def focal_loss(self, logits, targets):
        """Compute Focal Loss"""
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss)
        return focal_loss

    def asymmetric_loss(self, logits, targets):
        """Compute Asymmetric Loss"""
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos

        # Positive and negative loss terms
        los_pos = targets * torch.log(torch.clamp(xs_pos, min=1e-8)) * \
                  (1 - xs_pos).pow(self.gamma_pos)
        los_neg = (1 - targets) * torch.log(torch.clamp(xs_neg, min=1e-8)) * \
                  xs_pos.pow(self.gamma_neg)

        return -(los_pos + los_neg)

    def forward(self, logits, targets):
        # Get normalized loss weights
        weights = F.softmax(torch.stack([
            self.lambda_wbce,
            self.lambda_focal,
            self.lambda_asl
        ]), dim=0)

        if self.uncertainty_strategy == "U-MultiClass":
            # For multi-class uncertainty, use cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, 3),
                targets.view(-1, 3),
                reduction='none'
            )
            return loss.mean(), {'total': loss.mean()}

        elif self.uncertainty_strategy == "U-Ignore":
            # Create mask for uncertain labels
            mask = (targets != -1).float()

            # Compute losses only for certain labels
            wbce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='none'
            ) * mask

            focal_loss = self.focal_loss(logits, targets) * mask
            asl_loss = self.asymmetric_loss(logits, targets) * mask

            # Average over non-masked elements
            wbce_loss = wbce_loss.sum() / mask.sum().clamp(min=1)
            focal_loss = focal_loss.sum() / mask.sum().clamp(min=1)
            asl_loss = asl_loss.sum() / mask.sum().clamp(min=1)

        else:  # U-Zeros or U-Ones
            wbce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='mean'
            )
            focal_loss = self.focal_loss(logits, targets).mean()
            asl_loss = self.asymmetric_loss(logits, targets).mean()

        # Combine losses
        total_loss = (
                weights[0] * wbce_loss +
                weights[1] * focal_loss +
                weights[2] * asl_loss
        )

        loss_dict = {
            'wbce': wbce_loss.detach(),
            'focal': focal_loss.detach(),
            'asl': asl_loss.detach(),
            'total': total_loss.detach()
        }

        return total_loss, loss_dict

