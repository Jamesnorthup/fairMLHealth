# -*- coding: utf-8 -*-
"""
Tools producing reports of fairness, bias, or model performance measures

Contributors:
    camagallen <christine.allen@kensci.com>
"""
import aif360.sklearn.metrics as aif_mtrc
import fairlearn.metrics as fl_mtrc
from IPython.display import HTML
import logging
import pandas as pd
import numpy as np
import sklearn.metrics as sk_metric
import warnings

# Tutorial Libraries
from . import tutorial_helpers as helper


# Temporarily hide pandas SettingWithCopy warning
warnings.filterwarnings('ignore', module='pandas')
warnings.filterwarnings('ignore', module='sklearn')


__all__ = ["classification_fairness",
           "classification_performance",
           "regression_fairness",
           "regression_performance",
           "flag_suspicious"]


# TODO: reference this function in the tutorial rather than using explicit names
def get_report_labels(pred_type: str = "binary"):
    """ Returns a dictionary of category labels used by reporting functions

    Args:
        pred_type (b): number of classes in the prediction problem
    """
    valid_pred_types = ["binary", "multiclass", "regression"]
    if pred_type not in valid_pred_types:
        raise ValueError(f"pred_type must be one of {valid_pred_types}")
    c_note = "" if pred_type == "binary" else " (Weighted Avg)"
    report_labels = {'gf_label': "Group Fairness",
                     'if_label': "Individual Fairness",
                     'mp_label': f"Model Performance{c_note}",
                     'dt_label': "Data Metrics"
                     }
    return report_labels


def __format_fairtest_input(X, prtc_attr, y_true, y_pred, y_prob=None,
                            priv_grp=1):
    """ Formats data for use by fairness reporting functions.

    Args:
        X (array-like): Sample features
        prtc_attr (named array-like): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (1D array-like): Sample targets
        y_pred (1D array-like): Sample target predictions
        y_prob (1D array-like, optional): Sample target probabilities. Defaults
            to None.
        priv_grp (int, optional): label of the privileged group. Defaults
            to 1.

    Returns:
        Tuple containing formatted versions of all passed args.
    """
    __validate_report_inputs(X, prtc_attr, y_true, y_pred, y_prob, priv_grp)

    # Format inputs to required datatypes
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(prtc_attr, (np.ndarray, pd.Series)):
        if isinstance(prtc_attr, pd.Series):
            prtc_attr = pd.DataFrame(prtc_attr, columns=[prtc_attr.name])
        else:
            prtc_attr = pd.DataFrame(prtc_attr)
    if isinstance(y_true, (np.ndarray, pd.Series)):
        y_true = pd.DataFrame(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred)
    if isinstance(y_prob, np.ndarray):
        y_prob = pd.DataFrame(y_prob)
    for data in [y_true, y_pred, y_prob]:
        if data is not None and (len(data.shape) > 1 and data.shape[1] > 1):
            raise TypeError("Targets and predictions must be 1-Dimensional")

    # Format and set sensitive attributes as index for y dataframes
    pa_name = prtc_attr.columns.tolist()
    prtc_attr.reset_index(inplace=True, drop=True)
    y_true = pd.concat([prtc_attr, y_true.reset_index(drop=True)], axis=1
                       ).set_index(pa_name)
    y_pred = pd.concat([prtc_attr, y_pred.reset_index(drop=True)], axis=1
                       ).set_index(pa_name)
    y_pred.columns = y_true.columns
    if y_prob is not None:
        y_prob = pd.concat([prtc_attr, y_prob.reset_index(drop=True)],
                           axis=1
                           ).set_index(pa_name)
        y_prob.columns = y_true.columns

    # Ensure that protected attributes are integer-valued
    pa_cols = prtc_attr.columns.tolist()
    for c in pa_cols:
        binary_boolean = prtc_attr[c].isin([0, 1, False, True]).all()
        two_valued = (set(prtc_attr[c].astype(int)) == {0, 1})
        if not two_valued and binary_boolean:
            raise ValueError(
                    "prtc_attr must be binary or boolean and heterogeneous")
        prtc_attr.loc[:, c] = prtc_attr[c].astype(int)
        if isinstance(c, int):
            prtc_attr.rename(columns={c: f"prtc_attribute_{c}"}, inplace=True)

    return (X, prtc_attr, y_true, y_pred, y_prob)


def __binary_group_fairness_measures(X, prtc_attr, y_true, y_pred, y_prob=None,
                                     priv_grp=1):
    """ Returns a dictionary containing group fairness measures specific
        to binary classification problems

    Args:
        X (pandas DataFrame): Sample features
        prtc_attr (named array-like): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
        y_prob (pandas DataFrame, optional): Sample target probabilities.
            Defaults to None.
        priv_grp (int): Specifies which label indicates the privileged
                group. Defaults to 1.
    """
    pa_names = prtc_attr.columns.tolist()
    gf_vals = {}
    gf_vals['Statistical Parity Difference'] = \
        aif_mtrc.statistical_parity_difference(y_true, y_pred,
                                               prot_attr=pa_names)
    gf_vals['Disparate Impact Ratio'] = \
        aif_mtrc.disparate_impact_ratio(y_true, y_pred, prot_attr=pa_names)
    if not helper.is_tutorial_running() and not len(pa_names) > 1:
        gf_vals['Demographic Parity Difference'] = \
            fl_mtrc.demographic_parity_difference(y_true, y_pred,
                                                  sensitive_features=prtc_attr)
        gf_vals['Demographic Parity Ratio'] = \
            fl_mtrc.demographic_parity_ratio(y_true, y_pred,
                                             sensitive_features=prtc_attr)
    gf_vals['Average Odds Difference'] = \
        aif_mtrc.average_odds_difference(y_true, y_pred, prot_attr=pa_names)
    gf_vals['Equal Opportunity Difference'] = \
        aif_mtrc.equal_opportunity_difference(y_true, y_pred,
                                              prot_attr=pa_names)
    # Precision
    gf_vals['Positive Predictive Parity Difference'] = \
        aif_mtrc.difference(sk_metric.precision_score, y_true,
                            y_pred, prot_attr=pa_names, priv_group=priv_grp)
    gf_vals['Balanced Accuracy Difference'] = \
        aif_mtrc.difference(sk_metric.balanced_accuracy_score, y_true,
                            y_pred, prot_attr=pa_names, priv_group=priv_grp)
    # Add FairLearn metrics
    # ToDo: add deprecation warning and remove FairLearn from v0.1.0
    if not helper.is_tutorial_running() and not len(pa_names) > 1:
        gf_vals['Equalized Odds Difference'] = \
            fl_mtrc.equalized_odds_difference(y_true, y_pred,
                                              sensitive_features=prtc_attr)
        gf_vals['Equalized Odds Ratio'] = \
            fl_mtrc.equalized_odds_ratio(y_true, y_pred,
                                         sensitive_features=prtc_attr)
    # Add expanded metrics aid in understanding the above where tutorial is
    # not running
    if not helper.is_tutorial_running():
        gf_vals['Balanced Accuracy Ratio'] = \
            aif_mtrc.ratio(sk_metric.balanced_accuracy_score, y_true,
                           y_pred, prot_attr=pa_names, priv_group=priv_grp)
        # TPR
        gf_vals['Recall (TPR) Ratio'] = \
            aif_mtrc.ratio(sk_metric.recall_score, y_true,
                           y_pred, prot_attr=pa_names, priv_group=priv_grp)
        # FPR
        gf_vals['False Alarm (FPR) Ratio'] = \
            aif_mtrc.ratio(false_alarm_rate,
                           y_true, y_pred, prot_attr=pa_names,
                           priv_group=priv_grp)
        if y_prob is not None:
            gf_vals['AUC Difference'] = \
                aif_mtrc.difference(sk_metric.roc_auc_score, y_true, y_prob,
                                    prot_attr=pa_names, priv_group=priv_grp)
    return gf_vals


def __classification_performance_measures(y_true, y_pred):
    """ Returns a dictionary containing performance measures specific to
        classification problems

    Args:
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
    """
    # Generate a model performance report
    # If more than 2 classes, return the weighted average prediction scores
    n_class = y_true.append(y_pred).iloc[:, 0].nunique()
    target_labels = [f"target = {t}" for t in set(np.unique(y_true))]
    rprt = classification_performance(y_true.iloc[:, 0], y_pred.iloc[:, 0],
                                      target_labels)
    avg_lbl = "weighted avg" if n_class > 2 else target_labels[-1]
    #
    mp_vals = {}
    for score in ['precision', 'recall', 'f1-score']:
        mp_vals[score.title()] = rprt.loc[avg_lbl, score]
    mp_vals['Accuracy'] = rprt.loc['accuracy', 'accuracy']
    return mp_vals


def __data_metrics(y_true, priv_grp):
    """Returns a dictionary of data metrics applicable to evaluation of
    fairness

    Args:
        y_true (pandas DataFrame): Sample targets
        priv_grp (int): Specifies which label indicates the privileged
                group. Defaults to 1.
    """
    dt_vals = {}
    dt_vals['Prevalence of Privileged Class (%)'] = \
        round(100*y_true[y_true.eq(priv_grp)].sum()/y_true.shape[0])
    return dt_vals


def __individual_fairness_measures(X, prtc_attr, y_true, y_pred):
    """ Returns a dictionary of individual fairness measures for the data that
        were passed

    Args:
        X (pandas DataFrame): Sample features
        prtc_attr (named array-like): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
    """
    pa_names = prtc_attr.columns.tolist()
    # Generate dict of Individual Fairness measures
    if_vals = {}
    if_label = 'Individual Fairness'
    # consistency_score raises error if null values are present in X
    if X.notnull().all().all():
        if_vals['Consistency Score'] = \
            aif_mtrc.consistency_score(X, y_pred.iloc[:, 0])
    else:
        msg = "Cannot calculate consistency score. Null values present in data."
        logging.warning(msg)
    # Other aif360 metrics (not consistency) can handle null values
    if_vals['Between-Group Generalized Entropy Error'] = \
        aif_mtrc.between_group_generalized_entropy_error(y_true, y_pred,
                                                         prot_attr=pa_names)
    return if_vals


def __regres_group_fairness_measures(prtc_attr, y_true, y_pred, priv_grp=1):
    """ Returns a dictionary containing group fairness measures specific
        to regression problems

    Args:
        prtc_attr (named array-like): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
        priv_grp (int): Specifies which label indicates the privileged
                group. Defaults to 1.
    """
    pa_names = prtc_attr.columns.tolist()
    gf_vals = {}
    gf_vals['Statistical Parity Ratio'] = \
        fl_mtrc.statistical_parity_ratio(y_true, y_pred,
                                         prot_attr=prtc_attr)
    gf_vals['R2 Ratio'] = \
        aif_mtrc.ratio(sk_metric.r2_score, y_true, y_pred,
                       prot_attr=pa_names, priv_group=priv_grp)
    gf_vals['MAE Ratio'] = \
        aif_mtrc.ratio(sk_metric.mean_absolute_error, y_true, y_pred,
                       prot_attr=pa_names, priv_group=priv_grp)
    gf_vals['MSE Ratio'] = \
        aif_mtrc.ratio(sk_metric.mean_squared_error, y_true, y_pred,
                       prot_attr=pa_names, priv_group=priv_grp)
    return gf_vals


def __regression_performance_measures(y_true, y_pred):
    """ Returns a dictionary containing performance measures specific to
        classification problems

    Args:
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
    """
    mp_vals = {}
    report = regression_performance(y_true, y_pred)
    for row in report.iterrows():
        mp_vals[row[0]] = row[1]['Score']
    return mp_vals


def __validate_report_inputs(X, prtc_attr, y_true, y_pred, y_prob=None,
                             priv_grp=1):
    """ Raises error if data are of incorrect type or size for processing by
        the fairness or performance reporters

    Args:
        X (array-like): Sample features
        prtc_attr (array-like, named): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target predictions
        y_prob (array-like, 1-D): Sample target probabilities
    """
    valid_data_types = (pd.DataFrame, pd.Series, np.ndarray)
    for data in [X, prtc_attr, y_true, y_pred]:
        if not isinstance(data, valid_data_types):
            raise TypeError("input data is invalid type")
        if not data.shape[0] > 1:
            raise ValueError("input data is too small to measure")
    if y_prob is not None:
        if not isinstance(y_prob, valid_data_types):
            raise TypeError("y_prob is invalid type")
    if not isinstance(priv_grp, int):
        raise TypeError("priv_grp must be an integer")


def classification_fairness(X, prtc_attr, y_true, y_pred, y_prob=None,
                            priv_grp=1, sig_dec=4, **kwargs):
    """ Returns a pandas dataframe containing fairness measures for the model
        results

    Args:
        X (array-like): Sample features
        prtc_attr (array-like, named): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target predictions
        y_prob (array-like, 1-D): Sample target probabilities
        priv_grp (int): Specifies which label indicates the privileged
            group. Defaults to 1.
        sig_dec (int): number of significant decimals to which to round
            measure values. Defaults to 4.
    """
    # Validate and Format Arguments
    if not all([isinstance(a, int) for a in [priv_grp, sig_dec]]):
        raise ValueError(f"{a} must be an integer value")
    X, prtc_attr, y_true, y_pred, y_prob = \
        __format_fairtest_input(X, prtc_attr, y_true, y_pred, y_prob, priv_grp)

    # Temporarily prevent processing for more than 2 classes
    # ToDo: enable multiclass
    n_class = y_true.append(y_pred).iloc[:, 0].nunique()
    if n_class != 2:
        raise ValueError(
            "Reporter cannot yet process multiclass classification models")
    if n_class == 2:
        labels = get_report_labels()
    else:
        labels = get_report_labels("multiclass")
    gfl, ifl, mpl, dtl = labels.values()
    # Generate a dictionary of measure values to be converted t a dataframe
    mv_dict = {}
    mv_dict[gfl] = \
        __binary_group_fairness_measures(X, prtc_attr, y_true, y_pred,
                                         y_prob, priv_grp)
    mv_dict[ifl] = __individual_fairness_measures(X, prtc_attr, y_true, y_pred)
    mv_dict[dtl] = __data_metrics(y_true, priv_grp)
    if not kwargs.pop('skip_performance', False):
        mv_dict[mpl] = __classification_performance_measures(y_true, y_pred)
    # Convert scores to a formatted dataframe and return
    df = pd.DataFrame.from_dict(mv_dict, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df.loc[:, 'Value'] = df['Value'].astype(float).round(sig_dec)
    # Fix the order in which the metrics appear
    metric_order ={gfl: 0, ifl: 1, mpl: 2, dtl: 3}
    df.reset_index(inplace=True)
    df['sortorder'] = df['level_0'].map(metric_order)
    df = df.sort_values('sortorder').drop('sortorder', axis=1)
    df.set_index(['level_0', 'level_1'], inplace=True)
    df.rename_axis(('Metric', 'Measure'), inplace=True)
    return df


def classification_performance(y_true, y_pred, target_labels=None):
    """ Returns a pandas dataframe of the scikit-learn classification report,
        formatted for use in fairMLHealth tools

    Args:
        y_true (array): Target values. Must be compatible with model.predict().
        y_pred (array): Prediction values. Must be compatible with
            model.predict().
        target_labels (list of str): Optional labels for target values.
    """
    if target_labels is None:
        target_labels = [f"target = {t}" for t in set(y_true)]
    report = sk_metric.classification_report(y_true, y_pred, output_dict=True,
                                             target_names=target_labels)
    report = pd.DataFrame(report).transpose()
    # Move accuracy to separate row
    accuracy = report.loc['accuracy', :]
    report.drop('accuracy', inplace=True)
    report.loc['accuracy', 'accuracy'] = accuracy[0]
    return report


def binary_prediction_success(y_true, y_pred):
    """ Returns a dictionary with counts of TP, TN, FP, and FN
    """
    report = {}
    res = pd.concat((y_true, y_pred), axis=1)
    res.columns = ['t','p']
    report['TP'] = (res['t'].eq(1) & res['p'].eq(1)).sum()
    report['TN'] = (res['t'].eq(0) & res['p'].eq(0)).sum()
    report['FP'] = (res['t'].eq(0) & res['p'].eq(1)).sum()
    report['FN'] = (res['t'].eq(1) & res['p'].eq(0)).sum()
    return report


def flag_suspicious(df, caption="", as_styler=False):
    """ Generates embedded html pandas styler table containing a highlighted
        version of a model comparison dataframe

    Args:
        df (pandas dataframe): Model comparison dataframe (see)
        caption (str, optional): Optional caption for table. Defaults to "".
        as_styler (bool, optional): If True, returns a pandas Styler of the
            highlighted table (to which other styles/highlights can be added).
            Otherwise, returns the table as an embedded HTML object. Defaults
            to False .

    Returns:
        Embedded html or pandas.io.formats.style.Styler
    """
    if caption is None:
        caption = "Fairness Measures"
    #
    idx = pd.IndexSlice
    measures = df.index.get_level_values(1)
    ratios = df.loc[idx['Group Fairness',
                    [c.lower().endswith("ratio") for c in measures]], :].index
    difference = df.loc[idx['Group Fairness',
                        [c.lower().endswith("difference")
                         for c in measures]], :].index
    cs = df.loc[idx['Group Fairness',
                [c.lower().replace(" ", "_") == "consistency_score"
                 for c in measures]], :].index

    #
    def color_diff(row):
        clr = ['color:magenta'
               if (row.name in difference and not -0.1 < i < 0.1)
               else '' for i in row]
        return clr

    def color_if(row):
        clr = ['color:magenta'
               if (row.name in cs and i < 0.8) else '' for i in row]
        return clr

    def color_ratios(row):
        clr = ['color:magenta'
               if (row.name in ratios and not 0.8 < i < 1.2)
               else '' for i in row]
        return clr

    styled = df.style.set_caption(caption
                    ).apply(color_diff, axis=1
                    ).apply(color_ratios, axis=1
                    ).apply(color_if, axis=1)
    # Correct management of metric difference has yet to be determined for
    #   regression functions. Add style to o.o.r. difference for binary
    #   classification only
    if "MSE Ratio" not in measures:
        styled.apply(color_diff, axis=1)
    if as_styler:
        return styled
    else:
        return HTML(styled.render())


def regression_fairness(X, prtc_attr, y_true, y_pred, priv_grp=1, sig_dec=4):
    """ Returns a pandas dataframe containing fairness measures for the model
        results

    Args:
        X (array-like): Sample features
        prtc_attr (array-like, named): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target probabilities
        priv_grp (int): Specifies which label indicates the privileged
            group. Defaults to 1.
        sig_dec (int): number of significant decimals to which to round
            measure values. Defaults to 4.
    """
    # Validate and Format Arguments
    if not all([isinstance(a, int) for a in [priv_grp, sig_dec]]):
        raise ValueError(f"{a} must be an integer value")
    X, prtc_attr, y_true, y_pred, _ = \
        __format_fairtest_input(X, prtc_attr, y_true, y_pred, priv_grp)
    #
    gf_vals = \
        __regres_group_fairness_measures(prtc_attr, y_true, y_pred,
                                         priv_grp=priv_grp)
    #
    if_vals = __individual_fairness_measures(X, prtc_attr, y_true, y_pred)
    mp_vals = __regression_performance_measures(y_true, y_pred)
    dt_vals = __data_metrics(y_true, priv_grp)

    # Convert scores to a formatted dataframe and return
    labels = get_report_labels("regression")
    measures = {labels['gf_label']: gf_vals,
                labels['if_label']: if_vals,
                labels['mp_label']: mp_vals,
                labels['dt_label']: dt_vals}
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df.loc[:, 'Value'] = df['Value'].astype(float).round(sig_dec)
    return df


def regression_performance(y_true, y_pred):
    """ Returns a pandas dataframe of the regression performance metrics,
        similar to scikit's classification_performance

    Args:
        y_true (array): Target values. Must be compatible with model.predict().
        y_pred (array): Prediction values. Must be compatible with
            model.predict().
    """
    report = {}
    report['Rsqrd'] = sk_metric.r2_score(y_true, y_pred)
    report['MeanAE'] = sk_metric.mean_absolute_error(y_true, y_pred)
    report['MeanSE'] = sk_metric.mean_squared_error(y_true, y_pred)
    report = pd.DataFrame().from_dict(report, orient='index'
                          ).rename(columns={0: 'Score'})
    return report


'''
To File Before PR
'''
def sensitivity(y_true, y_pred):
    rprt = binary_prediction_success(y_true, y_pred)
    return rprt['TP']/(rprt['FN'] + rprt['TP'])

def false_alarm_rate(y_true, y_pred):
    rprt = binary_prediction_success(y_true, y_pred)
    return rprt['FP']/(rprt['FP'] + rprt['TN'])

def specificity(y_true, y_pred):
    rprt = binary_prediction_success(y_true, y_pred)
    return rprt['TN']/(rprt['FP'] + rprt['TN'])

def miss_rate(y_true, y_pred):
    rprt = binary_prediction_success(y_true, y_pred)
    return rprt['FN']/(rprt['FN'] + rprt['TP'])

# Feature that generates a RESULT1 table
'''
Generate a table of fairness metrics and stratified performance metrics for
    each specified feature

Requirements:
- Each feature must be discrete to run stratified analysis, and must be binary
to run the fairness assessment
-
'''