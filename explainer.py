import numpy as np
from sklearn.preprocessing import OrdinalEncoder

import shap
from lime import lime_tabular, lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm


class Explainer:
    def __init__(self, explainee):
        self.explainee = explainee

    def explain(self, dataset):
        raise NotImplemented


class SHAPExplainer(Explainer):
    def __init__(self, explainee, mode='default', index=None):
        super(SHAPExplainer, self).__init__(explainee)
        assert mode in ('default', 'tree', 'kernel', 'deep')
        self.mode = mode
        # only outputs importance of columns defined by INDEX
        self.index = index

    def explain(self, dataset, sample_size=50):
        if self.mode == 'default':
            explainer = shap.Explainer(self.explainee, dataset)
            result = explainer(dataset).values
        elif self.mode == 'tree':
            explainer = shap.TreeExplainer(self.explainee, dataset)
            result = np.vstack(explainer.shap_values(dataset))
        elif self.mode == 'kernel':
            explainer = shap.KernelExplainer(self.explainee, dataset)
            result = explainer.shap_values(dataset, silent=True)
        else:
            raise ValueError

        result = np.array(result)
        if self.index is not None:
            result = result[:, self.index]

        return result


class LIMEExplainer(Explainer):
    def __init__(self, explainee, num_classes, mode='tabular', **kwargs):
        super(LIMEExplainer, self).__init__(explainee)
        assert mode in ('tabular', 'image')
        self.mode = mode
        self.num_classes = num_classes
        self.kwargs = kwargs

    @staticmethod
    def exp_list2arr(lst):
        return np.array([v for k, v in lst])

    def explain(self, dataset):
        if self.mode == 'tabular':
            explainer = lime_tabular.LimeTabularExplainer(dataset, verbose=False, mode='regression',
                                                          discretize_continuous=False, **self.kwargs)
            result = []
            for x in dataset:
                result.append(self.exp_list2arr(
                    explainer.explain_instance(x, self.explainee.predict, num_features=dataset.shape[1]).as_list()
                ))

        elif self.mode == 'image':
            explainer = lime_image.LimeImageExplainer(verbose=False, **self.kwargs)
            segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

            result = []
            for x in dataset:
                exp = explainer.explain_instance(x[0], self.explainee, top_labels=10, hide_color=0,
                                                 num_samples=1000, segmentation_fn=segmenter)
                masks = []
                for i in range(self.num_classes):
                    _, mask = exp.get_image_and_mask(i, positive_only=False, num_features=10, hide_rest=False,
                                                     min_weight=0.01)
                    masks.append(mask)
                result.append(masks)

        else:
            raise ValueError

        return np.array(result)
