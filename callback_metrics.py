import keras
import sklearn.metrics as metrics
import numpy as np

class SegmentationMetricsLogger(keras.callbacks.Callback):
    def __init__(self, metrics, validation_gen, verbose=1):
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
        self.verbose = verbose
        
    
    def on_epoch_end(self, epoch, logs={}):
        """ Calculates the provided metric of every batch from valdiation_gen.
        """
        summations = { metric:0 for metric in self.metrics }

        # Calculate for every batch
        num_batches = self.validation_gen.__len__()
        for num in range(num_batches):
            x_val, y_val = self.validation_gen.__getitem__(num)
            y_pred = self.model.predict(x_val)

            # calculate tn, fn, tp, fp for every filter map in the batch.
            ct_values = self._calculate_ct_value_matrix(y_val, y_pred)
            
            # Calculate every metric and save mean per batch
            for metric in self.metrics:
                summations[metric] += self._available_metrics[metric](ct_values)

        # Calculate mean over batches for every metric
        for metric in summations.keys():
            summations[metric] /= num_batches
            self.metricHistories[metric].append(summations[metric])
        
        if self.verbose:
            print('Values for Epoch: ', summations)
            print('Metric History: ', self.metricHistories)

    def _batch_mean_metric(self, ct_values, func, starting_label=0):
        """ Calculates the mean for a metric over every colour channel and batch.
        """
        amount_labels = ct_values.shape[-1]
        batch_size = ct_values.shape[0]
        summation = 0

        for batch_num in range(batch_size):
            for label in range(starting_label, amount_labels):
                summation = summation + func(ct_values)

        return summation / (batch_size * (amount_labels - starting_label))
    
    def _calculate_ct_value_matrix(self, y_true, y_pred):
        """Calculates the confusion table values for every filter map in a batch.
        """
        print("Shape before: ", y_true.shape)
        ct_vals_matrix = np.empty((y_true.shape[0], y_true.shape[-1], 4))
        for b in range(y_true.shape[0]): # amount of batches
            for f in range(y_true.shape[-1]): # amount of segmentation maps
                ct_vals_matrix[b, f] = metrics.confusion_matrix(y_true[b,:,:,f].flatten(),
                                                                y_pred[b,:,:,f].flatten(),
                                                                labels=[0,1]).ravel()
        print("Shape after: ", ct_vals_matrix.shape)
        return ct_vals_matrix

    def _iou(self, ct_values):
        """ Calculates the IOU.
        
        # Arguments:
            ct_values: Tuple, containing tn, fp, fn, tp.
        
        # Returns:
            The metric.
        """
        tn, fp, fn, tp = ct_values
        union = tp + fn + fp
        return 1 if union is 0 else tp / union

    def _recall(self, ct_values):
        """ Calculates the Recall.
        
        # Arguments:
            ct_values: Tuple, containing tn, fp, fn, tp.
        
        # Returns:
            The metric.
        """
        tn, fp, fn, tp = ct_values

        denom = tp + fn
        return 0 if denom is 0 else tp / denom

    def _specificity(self, ct_values):
        """ Calculates the Specificity.
        
        # Arguments:
            ct_values: Tuple, containing tn, fp, fn, tp.
        
        # Returns:
            The metric.
        """
        tn, fp, fn, tp = ct_values

        denom = tn + fp
        return 1 if denom is 0 else tn / denom

    def _precision(self, ct_values):
        """ Calculates the Precision.
        
        # Arguments:
            ct_values: Tuple, containing tn, fp, fn, tp.
        
        # Returns:
            The metric.
        """
        tn, fp, fn, tp = ct_values

        denom = tp + fp
        return 1 if denom is 0 else tp / denom

    def _accuracy(self, ct_values):
        """ Calculates the Accuracy.
        
        # Arguments:
            ct_values: Tuple, containing tn, fp, fn, tp.
        
        # Returns:
            The metric.
        """
        tn, fp, fn, tp = ct_values

        return (tp + tn) / (tp + tn + fp + fn)

    def _balanced_accuracy(self, ct_values):
        """ Calculates the Balanced Accuracy.
        
        # Arguments:
            ct_values: Tuple, containing tn, fp, fn, tp.
        
        # Returns:
            The metric.
        """

        return (self._recall(ct_values) +
                self._specificity(ct_values) / 
                2)
    
    def _f1(self, ct_values):
        """ Calculates the F1 score.
        
        # Arguments:
            ct_values: Tuple, containing tn, fp, fn, tp.
        
        # Returns:
            The metric.
        """
        prec = self._precision(ct_values)
        rec = self._recall(ct_values)
        return 2 * (prec * rec / (prec + rec))

    def _mcc(self, ct_values):
        """ Calculates the mcc.
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        # Arguments:
            ct_values: Tuple, containing tn, fp, fn, tp.
        
        # Returns:
            The metric.
        """
        tn, fp, fn, tp = ct_values

        num = tp * tn - fp * fn
        den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5
    
        return num / den
         
    def _mIOU(self, ct_values):
        return self._batch_mean_metric(self._iou, ct_values)
    
    def _mRec(self, ct_values):
        return self._batch_mean_metric(self._recall, ct_values)
    
    def _mSpec(self, ct_values):
        return self._batch_mean_metric(self._specificity, ct_values)
    
    def _mPrec(self, ct_values):
        return self._batch_mean_metric(self._precision, ct_values)
    
    def _mAcc(self, ct_values):
        return self._batch_mean_metric(self._accuracy, ct_values)
    
    def _mBacc(self, ct_values):
        return self._batch_mean_metric(self._balanced_accuracy, ct_values)
    
    def _mF1(self, ct_values):
        return self._batch_mean_metric(self._f1, ct_values)
    
    def _mMcc(self, ct_values):
        return self._batch_mean_metric(self._mcc, ct_values)