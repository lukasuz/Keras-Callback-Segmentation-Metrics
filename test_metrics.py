import unittest
import numpy as np
import cv2
import metrics
import keras.backend as K
import segmentation_utils

class TestEvalMethods(unittest.TestCase):
    
    def setUp(self):
        self.width = 224
        self.height = 224
        self.b = [0, 0, 0]
        self.w = [255, 255, 255]
        self.colours = [self.b, self.w]

        self.path1 = './data/eyth_dataset/masks/vid4/frame25.png'
        self.path2 = './data/eyth_dataset/masks/vid4/frame30.png'

        self.mask1 = cv2.imread(self.path1, 1)
        self.mask1 = cv2.resize(self.mask1, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        self.mask1 = segmentation_utils.one_hot_image(self.mask1, self.colours)
        self.mask1_t = K.variable(self.mask1)

        self.mask2 = cv2.imread(self.path2, 1)
        self.mask2 = cv2.resize(self.mask2, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        self.mask2 = segmentation_utils.one_hot_image(self.mask2, self.colours)
        self.mask2_t = K.variable(self.mask2)

        self.matrix1 = np.array([[1, 1, 0],
                                 [1, 1, 0],
                                 [0, 0, 0]])

        self.matrix2 = np.array([[0, 0, 0],
                                 [0, 1, 1],
                                 [0, 1, 1]])

        self.matrix1_t = K.variable(self.matrix1)

        self.matrix2_t = K.variable(self.matrix2)

        self.ones = np.ones((3,3))
        self.zeros = np.zeros((3,3))

        self.ones_t = K.variable(self.ones)
        self.zeros_t = K.variable(self.zeros)

        self.col_matrix1 = np.array([[self.b, self.b, self.w],
                                     [self.b, self.b, self.w],
                                     [self.w, self.w, self.w]])

        self.col_matrix2 = np.array([[self.w, self.w, self.w],
                                     [self.w, self.b, self.b],
                                     [self.w, self.b, self.b]])

        self.y_true1 = np.zeros((200, 200, 2))
        self.y_pred1 = np.zeros((200, 200, 2))

        for x in range (100):
            for y in range(150):
                self.y_true1[x,y,1] = 1
        self.y_true1[:,:,0] = (self.y_true1[:,:,1] - 1) * -1

        for x in range (200):
            for y in range(100, 200):
                self.y_pred1[x,y,1] = 1
        self.y_pred1[:,:,0] = (self.y_pred1[:,:,1] - 1) * -1

        self.y_true1_t = K.variable(self.y_true1)
        self.y_pred1_t = K.variable(self.y_pred1)

        
    ### Basic operations ###

    def test_true_postives(self):
        # numpy arrays
        tp_rate = metrics._true_positives(self.matrix1, self.matrix2)
        self.assertEqual(tp_rate, 1)

        tp_rate = metrics._true_positives(self.matrix2, self.matrix2)
        self.assertEqual(tp_rate, 4)

        tp_rate = metrics._true_positives(self.ones, self.ones)
        self.assertEqual(tp_rate, 9)

        tp_rate = metrics._true_positives(self.zeros, self.zeros)
        self.assertEqual(tp_rate, 0)

        # tensors
        tp_rate = metrics._true_positives(self.matrix1_t, self.matrix2_t)
        self.assertTrue(K.eval(K.equal(tp_rate, K.variable(1, 'float32'))))

        tp_rate = metrics._true_positives(self.matrix2_t, self.matrix2_t)
        self.assertTrue(K.eval(K.equal(tp_rate, K.variable(4, 'float32'))))

        tp_rate = metrics._true_positives(self.ones_t, self.ones_t)
        self.assertTrue(K.eval(K.equal(tp_rate, K.variable(9, 'float32'))))

        tp_rate = metrics._true_positives(self.zeros_t, self.zeros_t)
        self.assertTrue(K.eval(K.equal(tp_rate, K.variable(0, 'float32'))))

    def test_true_negatives(self):
        # numpy arrays
        tn_rate = metrics._true_negatives(self.matrix1, self.matrix2)
        self.assertEqual(tn_rate, 2)

        tn_rate = metrics._true_negatives(self.matrix2, self.matrix2)
        self.assertEqual(tn_rate, 5)

        tn_rate = metrics._true_negatives(self.ones, self.ones)
        self.assertEqual(tn_rate, 0)

        tn_rate = metrics._true_negatives(self.zeros, self.zeros)
        self.assertEqual(tn_rate, 9)

        # tensors
        tn_rate = metrics._true_negatives(self.matrix1_t, self.matrix2_t)
        self.assertTrue(K.eval(K.equal(tn_rate, K.variable(2, 'float32'))))

        tn_rate = metrics._true_negatives(self.matrix2_t, self.matrix2_t)
        self.assertTrue(K.eval(K.equal(tn_rate, K.variable(5, 'float32'))))

        tn_rate = metrics._true_negatives(self.ones_t, self.ones_t)
        self.assertTrue(K.eval(K.equal(tn_rate, K.variable(0, 'float32'))))

        tn_rate = metrics._true_negatives(self.zeros_t, self.zeros_t)
        self.assertTrue(K.eval(K.equal(tn_rate, K.variable(9, 'float32'))))


    def test_false_positives(self):
        # numpy arrays
        fp_rate = metrics._false_positives(self.matrix1, self.matrix2)
        self.assertEqual(fp_rate, 3)

        fp_rate = metrics._false_positives(self.matrix2, self.matrix2)
        self.assertEqual(fp_rate, 0)

        fp_rate = metrics._false_positives(self.ones, self.ones)
        self.assertEqual(fp_rate, 0)

        fp_rate = metrics._false_positives(self.zeros, self.zeros)
        self.assertEqual(fp_rate, 0)

        fp_rate = metrics._false_positives(self.zeros, self.ones)
        self.assertEqual(fp_rate, 9)

        # tensors
        fp_rate = metrics._false_positives(self.matrix1_t, self.matrix2_t)
        self.assertTrue(K.eval(K.equal(fp_rate, K.variable(3, 'float32'))))

        fp_rate = metrics._false_positives(self.matrix2_t, self.matrix2_t)
        self.assertTrue(K.eval(K.equal(fp_rate, K.variable(0, 'float32'))))

        fp_rate = metrics._false_positives(self.ones_t, self.ones_t)
        self.assertTrue(K.eval(K.equal(fp_rate, K.variable(0, 'float32'))))

        fp_rate = metrics._false_positives(self.zeros_t, self.zeros_t)
        self.assertTrue(K.eval(K.equal(fp_rate, K.variable(0, 'float32'))))

        fp_rate = metrics._false_positives(self.zeros_t, self.ones_t)
        self.assertTrue(K.eval(K.equal(fp_rate, K.variable(9, 'float32'))))
    
    def test_false_negatives(self):
        fn_rate = metrics._false_negatives(self.matrix1, self.matrix2)
        self.assertEqual(fn_rate, 3)

        fn_rate = metrics._false_negatives(self.matrix2, self.matrix2)
        self.assertEqual(fn_rate, 0)

        fn_rate = metrics._false_negatives(self.ones, self.ones)
        self.assertEqual(fn_rate, 0)

        fn_rate = metrics._false_negatives(self.zeros, self.zeros)
        self.assertEqual(fn_rate, 0)

        fn_rate = metrics._false_negatives(self.zeros, self.ones)
        self.assertEqual(fn_rate, 0)

        fn_rate = metrics._false_negatives(self.ones, self.zeros)
        self.assertEqual(fn_rate, 9)

        # tensors
        fn_rate = metrics._false_negatives(self.matrix1_t, self.matrix2_t)
        self.assertTrue(K.eval(K.equal(fn_rate, K.variable(3, 'float32'))))

        fn_rate = metrics._false_negatives(self.matrix2_t, self.matrix2_t)
        self.assertTrue(K.eval(K.equal(fn_rate, K.variable(0, 'float32'))))

        fn_rate = metrics._false_negatives(self.ones_t, self.ones_t)
        self.assertTrue(K.eval(K.equal(fn_rate, K.variable(0, 'float32'))))

        fn_rate = metrics._false_negatives(self.zeros_t, self.zeros_t)
        self.assertTrue(K.eval(K.equal(fn_rate, K.variable(0, 'float32'))))

        fn_rate = metrics._false_negatives(self.zeros_t, self.ones_t)
        self.assertTrue(K.eval(K.equal(fn_rate, K.variable(0, 'float32'))))

        fn_rate = metrics._false_negatives(self.ones_t, self.zeros_t)
        self.assertTrue(K.eval(K.equal(fn_rate, K.variable(9, 'float32'))))

    ### Metrics ### 

    # def test_print_all_metrics(self):
    #     gt = self.mask1
    #     pred = self.mask2

    #     mIOU = metrics.mean_iou(gt, pred)
    #     mPrec = metrics.mean_precision(gt, pred)
    #     mRec = metrics.mean_recall(gt, pred)
    #     mAP = metrics.mean_average_precision(gt, pred)
    #     mf1 = metrics.mean_f1_score(gt, pred)
    #     mAcc = metrics.mean_accuracy(gt, pred)
    #     bAcc = metrics.balanced_accuracy(gt, pred)

    #     print("mIOU: ", mIOU)
    #     print("mPrec: ", mPrec)
    #     print("mRec: ", mRec)
    #     print("mAP: ", mAP)
    #     print("mf1: ", mf1)
    #     print("bAcc", bAcc)
    #     print("mAcc", mAcc)
        
    def test_iou(self):
        """ Test the IOU metric
        """
        # iou = metrics.mean_iou(self.mask2, self.mask1)
        # self.assertNotEqual(iou, 1.0)

        # iou = metrics.mean_iou(self.mask2_t, self.mask1_t)
        # self.assertTrue(K.eval(K.not_equal(iou, K.variable(1))))

        # iou = metrics.mean_iou(self.mask1, self.mask1)
        # self.assertEqual(iou, 1.0)

        # iou = metrics.mean_iou(self.mask1_t, self.mask1_t)
        # self.assertTrue(K.eval(K.equal(iou, K.variable(1))))

        # iou = metrics.mean_iou(self.mask1, self.mask1)
        # self.assertEqual(iou, 1.0)
        
        # gt_iou_score1 = ((24 * 224) / (224 * 224)) / 2
        # pred_out_score1 = metrics.mean_iou(
        #     self.own_mask_1, self.own_mask_2)

        # white_iou = (2400 / (224 * 224 - 24 * 200))
        # black_iou = (24 * 200) / (224 * 224 - 2 * 12 * 100)
        # gt_iou_score2 = (white_iou + black_iou) / 2

        # pred_out_score2 = metrics.mean_iou(
        #     self.own_mask_3, self.own_mask_4)

        # self.assertEqual(gt_iou_score1, pred_out_score1)
        # self.assertEqual(gt_iou_score2, pred_out_score2)

        # iou = metrics.mean_iou(self.y_true1, self.y_pred1)
        batch_1 = K.variable([self.mask1_t, self.mask1_t])
        batch_2 = K.variable([self.mask2_t, self.mask2_t])
        iou_t = metrics.mean_iou(batch_1, batch_2) # TODO WHY the same, but not while training
        iou_k = metrics.mean_iou(batch_1, batch_2)
        # iou_k = metrics.mean_iou_k(K.variable(self.y_true1), K.variable(self.y_pred1))
        # iou_c = metrics.iou_coef(K.variable(self.y_true1)[:,:,0], K.variable(self.y_pred1[:,:,0]))
        # print("IOU:", iou)
        print("IOU:", K.eval(iou_t))
        print("IOU_K:", K.eval(iou_k))
        
        # print("IOU_K:", K.eval(iou_k))
        # print("IOU_C:", K.eval(iou_c))


if __name__ == '__main__':
    unittest.main()