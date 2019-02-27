import keras
import sklearn.metrics as m
import numpy as np

class SegmentationMetricsLogger(keras.callbacks.Callback):
    def __init__(self, metrics, validation_gen):
        """ Creates a new metrics logger for semantic segmentation

        # Arguments:
            metrics: List containing the metrics that should be logged
            val_gen: Validation data generator the metrics should be calculated for,
                must be currently provided explicitly, because validation generators
                are not accessible in the callback functions. See:
                https://github.com/keras-team/keras/issues/10472
        """
        self._available_metrics = {
            'mIOU': self._mIOU,
            'mRec': self._mRec,
            'mSpec': self._mSpec,
            'mPrec': self._mPrec,
            'mAcc': self._mAcc,
            'mBacc': self._mBacc,
            'mF1': self._mF1,
            'mMcc': self._mMcc
        }

        for metric in metrics:
            if metric not in self._available_metrics.keys():
                raise ValueError('{0} not avaiable. Valid metrics are: {1}'.format(
                    metric, str(self._available_metrics.keys())
                ))

        super().__init__()
        self.metrics = metrics
        self.metricHistories = { metric:[] for metric in metrics }
        self.validation_gen = validation_gen
        
    
    def on_epoch_end(self, epoch, logs={}):
        """ Calculates the mean metric of every batch from valdiation_gen
        """
        summations = { metric:0 for metric in self.metrics }

        # Calculate for every batch
        num_batches = self.validation_gen.__len__()
        for num in range(num_batches):
            x_val, y_val = self.validation_gen.__getitem__(num)
            y_pred = self.model.predict(x_val)
            
            # Calc tp, tn, fp, fn
            # ct_values = m.confusion_matrix(y_val, y_pred).ravel()
            ct_values = None
            # Calculate every metric and save mean per batch
            for metric in self.metrics:
                summations[metric] += self._available_metrics[metric](y_val, y_pred, ct_values)

        # Calculate mean over batches for every metric
        for metric in summations.keys():
            summations[metric] /= num_batches
            self.metricHistories[metric].append(summations[metric])
        
        print('Values for Epoch: ', summations)
        print('Metric History: ', self.metricHistories)

    def _batch_mean_metric(self, y_true, y_pred, func, ct_values=None, starting_label=0):
        """ Calculates the mean for a metric over every colour channel and batch
        """
        amount_labels = y_pred.shape[-1]
        batch_size = y_pred.shape[0]
        summation = 0
        y_pred = np.round(y_pred)

        for batch_num in range(batch_size):
            for label in range(starting_label, amount_labels):
                summation = summation + func(y_true[batch_num,:,:,label].flatten(),
                                             y_pred[batch_num,:,:,label].flatten(),
                                             ct_values)

        return summation / (batch_size * (amount_labels - starting_label))

    def _loadCTValuesIfNone(self, y_true, y_pred, ct_values):
        if ct_values is None:
            return m.confusion_matrix(y_true, y_pred).ravel()
        else:
            return ct_values

    def _iou(self, y_true, y_pred, ct_values=None):
        tn, fp, fn, tp = self._loadCTValuesIfNone(y_true, y_pred, ct_values)
        union = tp + fn + fp
        return 1 if union is 0 else tp / union

    def _recall(self, y_true, y_pred, ct_values=None):
        tn, fp, fn, tp = self._loadCTValuesIfNone(y_true, y_pred, ct_values)

        denom = tp + fn
        return 1 if denom is 0 else tp / denom

    def _specificity(self, y_true, y_pred, ct_values=None):
        tn, fp, fn, tp = self._loadCTValuesIfNone(y_true, y_pred, ct_values)

        denom = tn + fp
        return 1 if denom is 0 else tn / denom

    def _precision(self, y_true, y_pred, ct_values=None):
        tn, fp, fn, tp = self._loadCTValuesIfNone(y_true, y_pred, ct_values)

        denom = tp + fp
        return 1 if denom is 0 else tp / denom

    def _accuracy(self, y_true, y_pred, ct_values=None):
        tn, fp, fn, tp = self._loadCTValuesIfNone(y_true, y_pred, ct_values)

        return (tp + tn) / (tp + tn + fp + fn)

    def _balanced_accuracy(self, y_true, y_pred, ct_values=None):
        ct_values = self._loadCTValuesIfNone(y_true, y_pred, ct_values)

        return (self._recall(y_true, y_pred, ct_values) +
                self._specificity(y_true, y_pred, ct_values) / 
                2)
    
    def _f1(self, y_true, y_pred, ct_values=None):
        self._loadCTValuesIfNone(y_true, y_pred, ct_values)

        prec = self._precision(y_true, y_pred, ct_values)
        rec = self._recall(y_true, y_pred, ct_values)
        return 2 * (prec * rec / (prec + rec))

    def _mcc(self, y_true, y_pred, ct_values=None):
        # mcc = https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        tn, fp, fn, tp = self._loadCTValuesIfNone(y_true, y_pred, ct_values)

        num = tp * tn - fp * fn
        den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5
    
        return num / den
         
    def _mIOU(self, y_true, y_pred, ct_values=None):
        return self._batch_mean_metric(y_true, y_pred, self._iou, ct_values)
    
    def _mRec(self, y_true, y_pred, ct_values=None):
        return self._batch_mean_metric(y_true, y_pred, self._recall, ct_values)
    
    def _mSpec(self, y_true, y_pred, ct_values=None):
        return self._batch_mean_metric(y_true, y_pred, self._specificity, ct_values)
    
    def _mPrec(self, y_true, y_pred, ct_values=None):
        return self._batch_mean_metric(y_true, y_pred, self._precision, ct_values)
    
    def _mAcc(self, y_true, y_pred, ct_values=None):
        return self._batch_mean_metric(y_true, y_pred, self._accuracy, ct_values)
    
    def _mBacc(self, y_true, y_pred, ct_values=None):
        return self._batch_mean_metric(y_true, y_pred, self._balanced_accuracy, ct_values)
    
    def _mF1(self, y_true, y_pred, ct_values=None):
        return self._batch_mean_metric(y_true, y_pred, self._f1, ct_values)
    
    def _mMcc(self, y_true, y_pred, ct_values=None):
        return self._batch_mean_metric(y_true, y_pred, self._mcc, ct_values)