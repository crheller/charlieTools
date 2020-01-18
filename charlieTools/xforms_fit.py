'''
Module w/ functions to fit an xforms model locally
'''
import nems
import nems.db as nd
import nems_lbhb.xform_wrappers as xfw
import nems.xform_helper as xhelp
import nems.xforms as xforms
from nems import get_setting
from nems.plugins import (default_keywords, default_loaders, default_fitters,
                          default_initializers)
from nems.registry import KeywordRegistry
import nems.gui.editors as gui
import io
import os
import logging
log = logging.getLogger(__name__)

def fit_xforms_model(batch, cellid, modelname, save_analysis=False):

    # parse modelname into loaders, modelspecs, and fit keys
    load_keywords, model_keywords, fit_keywords = modelname.split("_")

    # construct the meta data dict
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': load_keywords, 'fitkey': fit_keywords, 'modelspecname': model_keywords,
            'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': load_keywords}

    xforms_kwargs = {}
    xforms_init_context = {'cellid': cellid, 'batch': int(batch)}
    recording_uri = None
    kw_kwargs ={}

    xforms_lib = KeywordRegistry(**xforms_kwargs)

    xforms_lib.register_modules([default_loaders, default_fitters,
                                    default_initializers])
    xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

    keyword_lib = KeywordRegistry()
    keyword_lib.register_module(default_keywords)
    keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

    # Generate the xfspec, which defines the sequence of events
    # to run through (like a packaged-up script)
    xfspec = []

    # 0) set up initial context
    if xforms_init_context is None:
        xforms_init_context = {}
    if kw_kwargs is not None:
        xforms_init_context['kw_kwargs'] = kw_kwargs
    xforms_init_context['keywordstring'] = model_keywords
    xforms_init_context['meta'] = meta
    xfspec.append(['nems.xforms.init_context', xforms_init_context])

    # 1) Load the data
    xfspec.extend(xhelp._parse_kw_string(load_keywords, xforms_lib))

    # 2) generate a modelspec
    xfspec.append(['nems.xforms.init_from_keywords', {'registry': keyword_lib}])

    # 3) fit the data
    xfspec.extend(xhelp._parse_kw_string(fit_keywords, xforms_lib))

    # Generate a prediction
    xfspec.append(['nems.xforms.predict', {}])

    # 4) add some performance statistics
    xfspec.append(['nems.xforms.add_summary_statistics', {}])

    # 5) plot
    #xfspec.append(['nems_lbhb.lv_helpers.add_summary_statistics', {}])

    # Create a log stream set to the debug level; add it as a root log handler
    log_stream = io.StringIO()
    ch = logging.StreamHandler(log_stream)
    ch.setLevel(logging.DEBUG)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    ch.setFormatter(formatter)
    rootlogger = logging.getLogger()
    rootlogger.addHandler(ch)

    ctx = {}
    for xfa in xfspec:
        ctx = xforms.evaluate_step(xfa, ctx)

    # Close the log, remove the handler, and add the 'log' string to context
    log.info('Done (re-)evaluating xforms.')
    ch.close()
    rootlogger.removeFilter(ch)

    log_xf = log_stream.getvalue()


    modelspec = ctx['modelspec']
    if save_analysis:
        # save results
        if get_setting('USE_NEMS_BAPHY_API'):
            prefix = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT')) + '/results/'
        else:
            prefix = get_setting('NEMS_RESULTS_DIR')

        if type(cellid) is list:
            cell_name = cellid[0].split("-")[0]
        else:
            cell_name = cellid

        destination = os.path.join(prefix, str(batch), cell_name, modelspec.get_longname())

        modelspec.meta['modelpath'] = destination
        modelspec.meta.update(meta)

        log.info('Saving modelspec(s) to {0} ...'.format(destination))

        xforms.save_analysis(destination,
                            recording=ctx['rec'],
                            modelspec=modelspec,
                            xfspec=xfspec,
                            figures=[],
                            log=log_xf)

 
        # save performance and some other metadata in database Results table
        nd.update_results_table(modelspec)



    return ctx