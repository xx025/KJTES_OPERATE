import numpy as np
import torch

from modules.utils import points_to_matrix, revert_to_matrix, generate_mask_sizes, digital_recognition


class ChessBorard:
    def __init__(self,
                 det_model=None,
                 device=None,
                 ):
        self.TARGERT_SUM = 10
        self.template_digit_matrix = None
        self.template_position_matrix = None

        self.current_chessboard = None
        self.current_chessboard_positions = None

        self.W = 10
        self.H = 16

        self.ALL_NUM = self.W * self.H

        self.det_model = det_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.MASK_SIZES = generate_mask_sizes(
            W_N=self.W,
            H_N=self.H,
        )

        self.avgd = None
        self.center = (self.W / 2, self.H / 2)

    def recognize_chessboard(self, bord_image):
        xywh, digitals = digital_recognition(
            detmodel=self.det_model,
            image=bord_image,
            device=self.device
        )

        xycenter = xywh[:, :2] + xywh[:, 2:] / 2
        if len(digitals) == self.ALL_NUM:
            template_digit_matrix, template_position_matrix = points_to_matrix(
                [(x, y, int(d)) for (x, y), d in zip(xycenter, digitals)])
            h_template, w_template = bord_image.shape[:2]

            template_dots_matrix = template_position_matrix.astype(float)
            template_dots_matrix[..., 0] = template_dots_matrix[..., 0] / w_template  # x / 宽
            template_dots_matrix[..., 1] = template_dots_matrix[..., 1] / h_template  # y / 高
            self.template_digit_matrix = np.zeros_like(template_digit_matrix, dtype=int)
            self.template_position_matrix = template_dots_matrix
            self.current_chessboard = template_digit_matrix
            self.current_chessboard_positions = template_position_matrix
            self.avgd = np.mean(xywh[:, -2:])  # 记录平均大小

            print("Initialized template matrices from complete chessboard image.")
        else:
            if self.template_digit_matrix is None or self.template_position_matrix is None:
                raise ValueError(
                    "Template matrices are not initialized. Please provide a complete chessboard image first.")
            demo1_matrix, demo1_positions = revert_to_matrix(
                image=bord_image,
                xycenter=xycenter,
                digitals=digitals,
                template_digit_matrix=self.template_digit_matrix,
                template_dots_matrix=self.template_position_matrix
            )

            self.current_chessboard = demo1_matrix
            self.current_chessboard_positions = demo1_positions
        return self.current_chessboard, self.current_chessboard_positions


