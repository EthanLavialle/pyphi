#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib Plots for pyPhi 

Based on the work by Sal Garcia <sgarciam@ic.ac.uk> <salvadorgarciamunoz@gmail.com>

@author: Ethan Lavialle 
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyphi as phi
import pandas as pd
from datetime import datetime
from typing import Union, List, Tuple, Dict, Union, Optional


def timestr() -> str:
    """
    Returns the current date and time as a formatted string.

    Returns:
        str: Current date and time in 'YYYYMMDDHHMMSS' format.
    """
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")


def _create_classid_(df: pd.DataFrame, column: str, *, nbins: int = 5) -> pd.DataFrame:
    """
    Creates a CLASSID dataframe from values in a specified column by binning the data.

    Args:
        df (pd.DataFrame): The input dataframe.
        column (str): The name of the column to bin.
        nbins (int, optional): The number of bins to use. Default is 5.

    Returns:
        pd.DataFrame: A dataframe with binned CLASSID.
    """
    
    hist,bin_edges=np.histogram(df[column].values[np.logical_not(np.isnan(df[column].values))],bins=nbins )
    range_list=[]
    for i,e in enumerate(bin_edges[:-1]):
        range_list.append(str(np.round(bin_edges[i],3))+' to '+ str(np.round(bin_edges[i+1],3)))
    #range_list.append('NaN')
    bin_edges_=bin_edges.copy()
    bin_edges_[-1]=bin_edges_[-1]+0.1
    membership_=np.digitize(df[column].values,bin_edges_)
    membership=[]
    for m in membership_:
        membership.append(range_list[m-1])
    classid_df=df[df.columns[0]].to_frame()
    classid_df.insert(1,column,membership)
    return classid_df

def score_scatter(mvm_obj: dict, xydim: List[int], *,
                  CLASSID: bool = False, Xnew: Union[bool, np.ndarray, pd.DataFrame] = False,
                  rscores: bool = False, material: Union[bool, str] = False,
                  include_model: bool = False, title: str = '',
                  add_ci: bool = False, add_labels: bool = False, 
                  add_legend: bool = False, nbins: Union[bool, int] = False, 
                  marker_size: int = 4) -> None:
    """
    Creates a scatter plot of scores from a PLS/PCA model.

    Args:
        mvm_obj (dict): The PLS/PCA model object.
        xydim (List[int]): Dimensions to plot (e.g., [1, 2]).
        CLASSID (bool, optional): If True, includes class IDs. Default is False.
        Xnew (Union[bool, np.ndarray, pd.DataFrame], optional): New data to project. Default is False.
        rscores (bool, optional): If True, plots R-scores. Default is False.
        material (Union[bool, str], optional): Specifies material if applicable. Default is False.
        include_model (bool, optional): If True, includes the model data in the plot. Default is False.
        title (str, optional): Title of the plot. Default is ''.
        add_ci (bool, optional): If True, adds confidence intervals. Default is False.
        add_labels (bool, optional): If True, adds labels to points. Default is False.
        add_legend (bool, optional): If True, adds a legend to the plot. Default is False.
        nbins (Union[bool, int], optional): Number of bins for class ID creation. Default is False.
        marker_size (int, optional): Size of the scatter plot markers. Default is 4.

    Returns:
        None
    """
    
    if not(isinstance(nbins, bool)):
         if colorby in CLASSID.columns.to_list():
             classid_by_var = _create_classid_(CLASSID,colorby,nbins=nbins)
             CLASSID = classid_by_var.copy()


    mvm_obj=mvm_obj.copy()
    if ((mvm_obj['type']=='lpls') or  (mvm_obj['type']=='jrpls')  or  (mvm_obj['type']=='tpls')) and (not(isinstance(Xnew,bool))):    
        Xnew=False
        print('score scatter does not take Xnew for jrpls or lpls for now')
    
    if isinstance(Xnew,bool):
        
        if 'obsidX' in mvm_obj:
            ObsID_=mvm_obj['obsidX']
        else:
            ObsID_ = []
            for n in list(np.arange(mvm_obj['T'].shape[0])+1):
                ObsID_.append('Obs #'+str(n))  
        T_matrix=mvm_obj['T']    
        
        if not(rscores):        
            if (mvm_obj['type']=='lpls'):
                ObsID_=mvm_obj['obsidR']
            if (mvm_obj['type']=='jrpls') or (mvm_obj['type']=='tpls') :   
                 ObsID_=mvm_obj['obsidRi'][0]
        else:
            if (mvm_obj['type']=='lpls'):
                ObsID_=mvm_obj['obsidX']
                T_matrix=mvm_obj['Rscores']
            if (mvm_obj['type']=='jrpls') or (mvm_obj['type']=='tpls')  : 
                if isinstance(material,bool):
                    allobsids=[y for x in mvm_obj['obsidXi'] for y in x]
                    ObsID_=allobsids
                    clssid_obs=[]
                    clssid_class=[]
                    for i,R_ in enumerate(mvm_obj['Rscores']):
                        clssid_obs.extend(mvm_obj['obsidXi'][i])
                        clssid_class.extend([mvm_obj['materials'][i]]*len( mvm_obj['obsidXi'][i]))
                        if i==0:
                            allrscores=R_
                        else:
                            allrscores=np.vstack((allrscores,R_))
                    classid=pd.DataFrame(clssid_class,columns=['material'])
                    classid.insert(0,'obs',clssid_obs)
                    CLASSID=classid
                    colorby='material'
                    T_matrix=allrscores
                else:
                    ObsID_ = mvm_obj['obsidXi'][mvm_obj['materials'].index(material) ]
                    T_matrix = mvm_obj['Rscores'][mvm_obj['materials'].index(material) ]
    else:
        if isinstance(Xnew,np.ndarray):
            X_=Xnew.copy()
            ObsID_ = []
            for n in list(np.arange(Xnew.shape[0])+1):
                ObsID_.append('Obs #'+str(n))  
        elif isinstance(Xnew,pd.DataFrame):
            X_=np.array(Xnew.values[:,1:]).astype(float)
            ObsID_ = Xnew.values[:,0].astype(str)
            ObsID_ = ObsID_.tolist()
            
        if 'Q' in mvm_obj:  
            xpred=phi.pls_pred(X_,mvm_obj)
        else:
            xpred=phi.pca_pred(X_,mvm_obj)
        T_matrix=xpred['Tnew']
    
    if include_model:
        if 'obsidX' in mvm_obj:
            ObsID__=mvm_obj['obsidX'].copy()
        else:
            ObsID__ = []
            for n in list(np.arange(mvm_obj['T'].shape[0])+1):
                ObsID__.append('Model Obs #'+str(n))  
        T_matrix_=mvm_obj['T'].copy()    
      
        if isinstance(CLASSID,bool): # If there are no classids -> create create classids
            source=(['Model']*T_matrix_.shape[0])
            source.extend(['New']*T_matrix.shape[0])
            ObsID__.extend(ObsID_)
            CLASSID=pd.DataFrame.from_dict( {'ObsID':ObsID__,'_Source_':source })
            colorby='_Source_'
        else: #IF there are I need to augment it
            source=['Model']*T_matrix_.shape[0]
            CLASSID_=pd.DataFrame.from_dict( {CLASSID.columns[0]:ObsID__,colorby:source })
            ObsID__.extend(ObsID_)
            CLASSID = pd.concat([CLASSID_,CLASSID])
        ObsID_=ObsID__.copy()    
        T_matrix=np.vstack((T_matrix_,T_matrix ))

    ObsNum_=[]    
    for n in list(range(1,len(ObsID_)+1)):
                ObsNum_.append(str(n)) 

    if isinstance(CLASSID,bool): # No CLASSIDS
     
        x_=T_matrix[:,[xydim[0]-1]]
        y_=T_matrix[:,[xydim[1]-1]]
        source = {'x': x_, 'y': y_, 'ObsID': ObsID_, 'ObsNum' : ObsNum_}
        
    
        # Create the figure and axis
        fig, ax = plt.subplots()
        sc = ax.scatter(source['x'], source['y'], s=marker_size)

        if add_ci:
            T_aux1=mvm_obj['T'][:,[xydim[0]-1]]
            T_aux2=mvm_obj['T'][:,[xydim[1]-1]]
            T_aux = np.hstack((T_aux1,T_aux2))
            st=(T_aux.T @ T_aux)/T_aux.shape[0]
            [xd95,xd99,yd95p,yd95n,yd99p,yd99n]=phi.scores_conf_int_calc(st,mvm_obj['T'].shape[0])
            
            ax.plot(xd95, yd95p, 'gold', linestyle='dashed')
            ax.plot(xd95, yd95n, 'gold', linestyle='dashed')
            ax.plot(xd99, yd99p, 'red', linestyle='dashed')
            ax.plot(xd99, yd99n, 'red', linestyle='dashed')

        if add_labels:
            for i, txt in enumerate(source['ObsID']):
                ax.annotate(txt, (source['x'][i], source['y'][i]), xytext=(5, 5), textcoords='offset points')

        # Set axis labels
        if not rscores:
            ax.set_xlabel(f't [{xydim[0]}]')
            ax.set_ylabel(f't [{xydim[1]}]')
        else:
            ax.set_xlabel(f'r [{xydim[0]}]')
            ax.set_ylabel(f'r [{xydim[1]}]')

        # Add vertical and horizontal lines
        ax.axhline(y=0, color='black', linewidth=2)
        ax.axvline(x=0, color='black', linewidth=2)

        # Add title
        ax.set_title(f'Score Scatter t[{xydim[0]}] - t[{xydim[1]}] {title}')
        
        # Live annotation of points
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def scatter_update_annot(ind):
    
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "Obs#: {} \n{}".format(" ".join([source['ObsNum'][n] for n in ind["ind"]]),
                                       " ".join([source['ObsID'][n] for n in ind["ind"]]))
            annot.set_text(text)
            #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(0.4)
    

        def scatter_hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    scatter_update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", scatter_hover)


        # Show the plot
        plt.show()


    else: # YES CLASSIDS
        Classes_ = CLASSID['Class'].unique()
        A = len(Classes_)
        colormap = plt.get_cmap('rainbow')
        color_mapping = colormap(np.linspace(0, 1, A))

        if Classes_[0] == 'Model':
            color_mapping = np.vstack((np.array([225/255, 225/255, 225/255, 1]), color_mapping))

        # Create the figure and axis
        fig, ax = plt.subplots()

        # Plot the scatter points by class
        for classid_in_turn in Classes_:
            x_aux = []
            y_aux = []
            obsid_aux = []
            obsnum_aux = []
            classid_aux = []
            
            for i in range(len(ObsID_)):
                if CLASSID['Class'][i] == classid_in_turn:
                    x_aux.append(T_matrix[i, xydim[0] - 1])
                    y_aux.append(T_matrix[i, xydim[1] - 1])
                    obsid_aux.append(ObsID_[i])
                    obsnum_aux.append(ObsNum_[i])
                    classid_aux.append(classid_in_turn)
            
            color_ = color_mapping[Classes_.tolist().index(classid_in_turn)]
            scatter = ax.scatter(x_aux, y_aux, color=color_, label=classid_in_turn)
            
            if add_labels:
                for i, txt in enumerate(obsid_aux):
                    ax.annotate(txt, (x_aux[i], y_aux[i]), xytext=(5, 5), textcoords='offset points')

        # Add confidence intervals if add_ci is True
        if add_ci:
            T_aux1 = mvm_obj['T'][:, [xydim[0] - 1]]
            T_aux2 = mvm_obj['T'][:, [xydim[1] - 1]]
            T_aux = np.hstack((T_aux1, T_aux2))
            st = (T_aux.T @ T_aux) / T_aux.shape[0]
            xd95, xd99, yd95p, yd95n, yd99p, yd99n = phi.scores_conf_int_calc(st, mvm_obj['T'].shape[0])
            
            ax.plot(xd95, yd95p, 'gold', linestyle='dashed')
            ax.plot(xd95, yd95n, 'gold', linestyle='dashed')
            ax.plot(xd99, yd99p, 'red', linestyle='dashed')
            ax.plot(xd99, yd99n, 'red', linestyle='dashed')

        # Set axis labels
        if not rscores:
            ax.set_xlabel(f't [{xydim[0]}]')
            ax.set_ylabel(f't [{xydim[1]}]')
        else:
            ax.set_xlabel(f'r [{xydim[0]}]')
            ax.set_ylabel(f'r [{xydim[1]}]')

        # Add vertical and horizontal lines
        ax.axhline(y=0, color='black', linewidth=2)
        ax.axvline(x=0, color='black', linewidth=2)

        # Add title
        ax.set_title(f'Score Scatter t[{xydim[0]}] - t[{xydim[1]}] {title}')

        # Add legend if add_legend is True
        if add_legend:
            ax.legend()

        # Show the plot
        plt.show()


import matplotlib.pyplot as plt
import numpy as np

def loadings(mvm_obj: dict, plotwidth: int = 10, xgrid: bool = False, 
             addtitle: str = '', material: Union[bool, str] = False, 
             zspace: bool = False) -> None:
    """
    Plots the loadings of a PLS/PCA model.

    Args:
        mvm_obj (dict): The PLS/PCA model object.
        plotwidth (int, optional): Width of the plot. Default is 10.
        xgrid (bool, optional): If True, adds grid lines to the x-axis. Default is False.
        addtitle (str, optional): Additional title for the plot. Default is ''.
        material (Union[bool, str], optional): Specifies material if applicable. Default is False.
        zspace (bool, optional): If True, uses Z-space loadings. Default is False.

    Returns:
        None
    """
    
    mvm_obj = mvm_obj.copy()
    space_lbl = 'X'
    A = mvm_obj['T'].shape[1]
    num_varX = mvm_obj['P'].shape[0] if 'P' in mvm_obj else 0
    is_pls = 'Q' in mvm_obj
    
    if mvm_obj['type'] in ['lpls', 'jrpls', 'tpls']:
        loading_lbl = 'S*'
        if mvm_obj['type'] == 'lpls':
            mvm_obj['Ws'] = mvm_obj['Ss']
        if isinstance(material, bool) and not zspace:
            mvm_obj['Ws'] = mvm_obj['Ss']
        if mvm_obj['type'] in ['jrpls', 'tpls'] and not isinstance(material, bool):
            mvm_obj['Ws'] = mvm_obj['Ssi'][mvm_obj['materials'].index(material)]
            mvm_obj['varidX'] = mvm_obj['varidXi'][mvm_obj['materials'].index(material)]
        elif mvm_obj['type'] == 'tpls' and zspace:
            mvm_obj['varidX'] = mvm_obj['varidZ']
            loading_lbl = 'Wz*'
            space_lbl = 'Z'
    else:
        loading_lbl = 'W*'

    lv_prefix = 'LV #' if is_pls else 'PC #'
    lv_labels = [lv_prefix + str(a+1) for a in range(A)]
    
    XVar = mvm_obj.get('varidX', ['XVar #' + str(n+1) for n in range(num_varX)])
    if is_pls:
        YVar = mvm_obj.get('varidY', ['YVar #' + str(n+1) for n in range(mvm_obj['Q'].shape[0])])
    
    # Plot X space loadings
    fig, ax = plt.subplots(figsize=(plotwidth, 5))
    width = 0.8 / A  # width of bars
    indices = np.arange(len(XVar))

    for i in range(A):
        ax.bar(indices + i * width, mvm_obj['Ws'][:, i] if is_pls else mvm_obj['P'][:, i], width=width, label=f"{lv_labels[i]}")
    
    ax.set_title(f"{space_lbl} Space Loadings{addtitle}")
    ax.set_xlabel('Variables')
    ax.set_ylabel(f"{loading_lbl}")
    ax.axhline(0, color='black', linewidth=2)
    if xgrid:
        ax.xaxis.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
    else:
        ax.xaxis.grid(False)
    ax.set_xticks(indices + width * (A-1) / 2)
    ax.set_xticklabels(XVar, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot Y space loadings if PLS model
    if is_pls:
        fig, ax = plt.subplots(figsize=(plotwidth, 5))
        indices = np.arange(len(YVar))

        for i in range(A):
            ax.bar(indices + i * width, mvm_obj['Q'][:, i], width=width, label=f"{lv_labels[i]}")
        
        ax.set_title(f"Y Space Loadings{addtitle}")
        ax.set_xlabel('Variables')
        ax.set_ylabel(f"Q")
        ax.axhline(0, color='black', linewidth=2)
        if xgrid:
            ax.xaxis.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
        else:
            ax.xaxis.grid(False)
        ax.set_xticks(indices + width * (A-1) / 2)
        ax.set_xticklabels(YVar, rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.show()




def loadings_map(mvm_obj: dict, dims: List[int], *, plotwidth: int = 8, addtitle: str = '', 
                 material: Union[bool, str] = False, zspace: bool = False, 
                 textalpha: float = 0.75) -> None:
    """
    Plots the loadings map for a PLS/PCA model.

    Args:
        mvm_obj (dict): The PLS/PCA model object.
        dims (List[int]): Dimensions to plot (e.g., [1, 2]).
        plotwidth (int, optional): Width of the plot. Default is 8.
        addtitle (str, optional): Additional title for the plot. Default is ''.
        material (Union[bool, str], optional): Specifies material if applicable. Default is False.
        zspace (bool, optional): If True, uses Z-space loadings. Default is False.
        textalpha (float, optional): Transparency of the text annotations. Default is 0.75.

    Returns:
        None
    """

    mvm_obj = mvm_obj.copy()
    A = mvm_obj['T'].shape[1]
    if (mvm_obj['type'] == 'lpls') or (mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls'):
        if mvm_obj['type'] == 'lpls':
            mvm_obj['Ws'] = mvm_obj['Ss']
        if isinstance(material, bool) and not zspace:
            mvm_obj['Ws'] = mvm_obj['Ss']
        if ((mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls')) and not isinstance(material, bool):
            mvm_obj['Ws'] = mvm_obj['Ssi'][mvm_obj['materials'].index(material)]
            mvm_obj['varidX'] = mvm_obj['varidXi'][mvm_obj['materials'].index(material)]
        elif (mvm_obj['type'] == 'tpls') and zspace:
            mvm_obj['varidX'] = mvm_obj['varidZ']
    else:
        num_varX = mvm_obj['P'].shape[0]

    if 'Q' in mvm_obj:
        lv_prefix = 'LV #'
        lv_labels = [lv_prefix + str(a + 1) for a in range(A)]

        if 'varidX' in mvm_obj:
            XVar = mvm_obj['varidX']
        else:
            XVar = ['XVar #' + str(n + 1) for n in range(num_varX)]

        num_varY = mvm_obj['Q'].shape[0]
        if 'varidY' in mvm_obj:
            YVar = mvm_obj['varidY']
        else:
            YVar = ['YVar #' + str(n + 1) for n in range(num_varY)]

        x_ws = mvm_obj['Ws'][:, dims[0] - 1] / np.max(np.abs(mvm_obj['Ws'][:, dims[0] - 1]))
        y_ws = mvm_obj['Ws'][:, dims[1] - 1] / np.max(np.abs(mvm_obj['Ws'][:, dims[1] - 1]))

        x_q = mvm_obj['Q'][:, dims[0] - 1] / np.max(np.abs(mvm_obj['Q'][:, dims[0] - 1]))
        y_q = mvm_obj['Q'][:, dims[1] - 1] / np.max(np.abs(mvm_obj['Q'][:, dims[1] - 1]))

        fig, ax = plt.subplots(figsize=(plotwidth, plotwidth))
        ax.scatter(x_ws, y_ws, color='darkblue', label='X Loadings')
        ax.scatter(x_q, y_q, color='red', label='Y Loadings')

        for i, txt in enumerate(XVar):
            ax.annotate(txt, (x_ws[i], y_ws[i]), color='darkgray', alpha=textalpha)
        for i, txt in enumerate(YVar):
            ax.annotate(txt, (x_q[i], y_q[i]), color='darkgray', alpha=textalpha)

        ax.axhline(0, color='black', linewidth=2)
        ax.axvline(0, color='black', linewidth=2)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        ax.set_xlabel(lv_labels[dims[0] - 1])
        ax.set_ylabel(lv_labels[dims[1] - 1])
        ax.set_title(f"Loadings Map LV[{dims[0]}] - LV[{dims[1]}] {addtitle}")

        ax.legend()
        plt.show()
    else:
        lv_prefix = 'PC #'
        lv_labels = [lv_prefix + str(a + 1) for a in range(A)]

        if 'varidX' in mvm_obj:
            XVar = mvm_obj['varidX']
        else:
            XVar = ['XVar #' + str(n + 1) for n in range(num_varX)]

        x_p = mvm_obj['P'][:, dims[0] - 1]
        y_p = mvm_obj['P'][:, dims[1] - 1]

        fig, ax = plt.subplots(figsize=(plotwidth, plotwidth))
        ax.scatter(x_p, y_p, color='darkblue', label='X Loadings')

        for i, txt in enumerate(XVar):
            ax.annotate(txt, (x_p[i], y_p[i]), color='darkgray', alpha=textalpha)

        ax.axhline(0, color='black', linewidth=2)
        ax.axvline(0, color='black', linewidth=2)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        ax.set_xlabel(lv_labels[dims[0] - 1])
        ax.set_ylabel(lv_labels[dims[1] - 1])
        ax.set_title(f"Loadings Map PC[{dims[0]}] - PC[{dims[1]}] {addtitle}")

        ax.legend()
        plt.show()


def r2pv(mvm_obj: Dict, *, plotwidth: int = 8, plotheight: int = 6, addtitle: str = '', material: Union[bool, str] = False, zspace: bool = False) -> None:
    """
    Generates a bar plot for R2 Per Variable (R2PV) for the given multivariate model object.

    Args:
    - mvm_obj (Dict): The multivariate model object.
    - plotwidth (int, optional): The width of the plot. Defaults to 8.
    - plotheight (int, optional): The height of the plot. Defaults to 6.
    - addtitle (str, optional): Additional title for the plot. Defaults to ''.
    - material (Union[bool, str], optional): Material specification for the plot. Defaults to False.
    - zspace (bool, optional): Flag to indicate Z-space in the plot. Defaults to False.

    Returns:
        None
    """

    A = mvm_obj['T'].shape[1]
    yaxlbl = 'X'
    if (mvm_obj['type'] == 'lpls') or (mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls'):
        if ((mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls')) and not(isinstance(material, bool)):
            mvm_obj['r2xpv'] = mvm_obj['r2xpvi'][mvm_obj['materials'].index(material)]
            mvm_obj['varidX'] = mvm_obj['varidXi'][mvm_obj['materials'].index(material)]
        elif (mvm_obj['type'] == 'tpls') and zspace:
            mvm_obj['r2xpv'] = mvm_obj['r2zpv']
            mvm_obj['varidX'] = mvm_obj['varidZ']
            yaxlbl = 'Z'
    else:
        num_varX = mvm_obj['P'].shape[0]

    is_pls = 'Q' in mvm_obj
    lv_prefix = 'LV #' if is_pls else 'PC #'
        
    lv_labels = [lv_prefix + str(a + 1) for a in range(A)]

    if 'varidX' in mvm_obj:
        XVar = mvm_obj['varidX']
    else:
        XVar = ['XVar #' + str(n + 1) for n in range(num_varX)]

    r2pvX_dict = {'XVar': XVar}
    for i in range(A):
        r2pvX_dict.update({lv_labels[i]: mvm_obj['r2xpv'][:, i].tolist()})

    if is_pls:
        num_varY = mvm_obj['Q'].shape[0]
        if 'varidY' in mvm_obj:
            YVar = mvm_obj['varidY']
        else:
            YVar = ['YVar #' + str(n + 1) for n in range(num_varY)]
        r2pvY_dict = {'YVar': YVar}
        for i in range(A):
            r2pvY_dict.update({lv_labels[i]: mvm_obj['r2ypv'][:, i].tolist()})
    
    colormap = plt.cm.rainbow
    colors = colormap(np.linspace(0, 1, A))

    # Plot R2X Per Variable
    fig, ax = plt.subplots(figsize=(plotwidth, plotheight))
    bottom = np.zeros(len(XVar))
    for i, label in enumerate(lv_labels):
        ax.bar(XVar, r2pvX_dict[label], label=label, color=colors[i], bottom=bottom)
        bottom += np.array(r2pvX_dict[label])
    ax.set_title(f"R2{yaxlbl} Per Variable {addtitle}")
    ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    ax.set_xlabel('Variables')
    ax.set_ylabel(f'R2{yaxlbl}')
    ax.legend()
    # plt.xticks(rotation=45)
    plt.show()

    if is_pls:
        # Plot R2Y Per Variable
        fig, ax = plt.subplots(figsize=(plotwidth, plotheight))
        bottom = np.zeros(len(YVar))
        for i, label in enumerate(lv_labels):
            ax.bar(YVar, r2pvY_dict[label], label=label, color=colors[i], bottom=bottom)
            bottom += np.array(r2pvY_dict[label])
        ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
        ax.set_title(f"R2Y Per Variable {addtitle}")
        ax.set_xlabel('Variables')
        ax.set_ylabel('R2Y')
        ax.legend()
        # plt.xticks(rotation=45)
        plt.show()



def r2(mvm_obj: Dict, *, plotwidth: int = 8, plotheight: int = 6, addtitle: str = '', material: Union[bool, str] = False, zspace: bool = False) -> None:
    """
    Generates a bar plot for R2 for the given multivariate model object.

    Args:
    - mvm_obj (Dict): The multivariate model object.
    - plotwidth (int, optional): The width of the plot. Defaults to 8.
    - plotheight (int, optional): The height of the plot. Defaults to 6.
    - addtitle (str, optional): Additional title for the plot. Defaults to ''.
    - material (Union[bool, str], optional): Material specification for the plot. Defaults to False.
    - zspace (bool, optional): Flag to indicate Z-space in the plot. Defaults to False.

    Returns:
        None
    """

    mvm_obj = mvm_obj.copy()
    A = mvm_obj['T'].shape[1]
    yaxlbl = 'X'
    if (mvm_obj['type'] == 'lpls') or (mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls'):
        if ((mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls')) and not(isinstance(material, bool)):
            mvm_obj['r2xpv'] = mvm_obj['r2xpvi'][mvm_obj['materials'].index(material)]
            mvm_obj['varidX'] = mvm_obj['varidXi'][mvm_obj['materials'].index(material)]
        elif (mvm_obj['type'] == 'tpls') and zspace:
            mvm_obj['r2xpv'] = mvm_obj['r2zpv']
            mvm_obj['varidX'] = mvm_obj['varidZ']
            yaxlbl = 'Z'
    else:
        num_varX = mvm_obj['P'].shape[0]

    is_pls = 'Q' in mvm_obj
    lv_prefix = 'LV #' if is_pls else 'PC #'
        
    lv_labels = [lv_prefix + str(a + 1) for a in range(A)]

    if 'varidX' in mvm_obj:
        XVar = mvm_obj['varidX']
    else:
        XVar = ['XVar #' + str(n + 1) for n in range(num_varX)]

    # R2X values across variables for each component
    r2X_dict = {}
    for i in range(A):
        # Update the dictionary with the R2X values for the current component
        r2X_dict[lv_labels[i]] = mvm_obj['r2x'][i].tolist()

    if is_pls:
        num_varY = mvm_obj['Q'].shape[0]
        if 'varidY' in mvm_obj:
            YVar = mvm_obj['varidY']
        else:
            YVar = ['YVar #' + str(n + 1) for n in range(num_varY)]
        # R2Y values across variables for each component
        r2Y_dict = {}
        for i in range(A):
            # Update the dictionary with the R2Y values for the current component
            r2Y_dict[lv_labels[i]] = mvm_obj['r2y'][i].tolist()
    
    colormap = plt.cm.rainbow
    colors = colormap(np.linspace(0, 1, A))

    # Convert r2X_dict to lists for plotting
    components = list(r2X_dict.keys())
    r2_values = list(r2X_dict.values())

    # Plot R2X
    fig, ax = plt.subplots(figsize=(plotwidth, plotheight))
    ax.bar(components, r2_values, color=colors)  
    ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    ax.set_title(f"R2{yaxlbl} {addtitle}")
    ax.set_xlabel('Components')
    ax.set_ylabel(f'R2{yaxlbl}')
    plt.show()

    if is_pls:
        # Plot R2Y
        r2y_values = list(r2Y_dict.values())
        fig, ax = plt.subplots(figsize=(plotwidth, plotheight))
        ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
        ax.bar(lv_labels, r2y_values, color=colors)
        ax.set_title(f"R2Y {addtitle}")
        ax.set_xlabel('Components')
        ax.set_ylabel('R2Y')
        plt.show()


def r2c(mvm_obj: Dict, *, plotwidth: int = 8, plotheight: int = 6, addtitle: str = '', material: Union[bool, str] = False, zspace: bool = False) -> None:
    """
    Generates a cumulative R2 bar plot for the given multivariate model object.

    Args:
    - mvm_obj (Dict): The multivariate model object.
    - plotwidth (int, optional): The width of the plot. Defaults to 8.
    - plotheight (int, optional): The height of the plot. Defaults to 6.
    - addtitle (str, optional): Additional title for the plot. Defaults to ''.
    - material (Union[bool, str], optional): Material specification for the plot. Defaults to False.
    - zspace (bool, optional): Flag to indicate Z-space in the plot. Defaults to False.

    Returns:
        None
    """

    mvm_obj = mvm_obj.copy()
    A = mvm_obj['T'].shape[1]
    yaxlbl = 'X'
    if (mvm_obj['type'] == 'lpls') or (mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls'):
        if ((mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls')) and not(isinstance(material, bool)):
            mvm_obj['r2xpv'] = mvm_obj['r2xpvi'][mvm_obj['materials'].index(material)]
            mvm_obj['varidX'] = mvm_obj['varidXi'][mvm_obj['materials'].index(material)]
        elif (mvm_obj['type'] == 'tpls') and zspace:
            mvm_obj['r2xpv'] = mvm_obj['r2zpv']
            mvm_obj['varidX'] = mvm_obj['varidZ']
            yaxlbl = 'Z'
    else:
        num_varX = mvm_obj['P'].shape[0]

    is_pls = 'Q' in mvm_obj
    lv_prefix = 'LV #' if is_pls else 'PC #'
        
    lv_labels = [lv_prefix + str(a + 1) for a in range(A)]

    # R2X values across variables for each component
    r2X_dict = {}
    for i in range(A):
        # Update the dictionary with the R2X values for the current component
        r2X_dict[lv_labels[i]] = mvm_obj['r2x'][i].tolist()

    if is_pls:
        num_varY = mvm_obj['Q'].shape[0]
        if 'varidY' in mvm_obj:
            YVar = mvm_obj['varidY']
        else:
            YVar = ['YVar #' + str(n + 1) for n in range(num_varY)]
        # R2Y values across variables for each component
        r2Y_dict = {}
        for i in range(A):
            # Update the dictionary with the R2Y values for the current component
            r2Y_dict[lv_labels[i]] = mvm_obj['r2y'][i].tolist()

    # Convert r2X_dict to lists for plotting
    components = list(r2X_dict.keys())
    r2_values = list(r2X_dict.values())

    r2x_cumulative = np.cumsum(r2_values)
    

    # Plot Cumulative R2X as bar chart
    fig, ax = plt.subplots(figsize=(plotwidth, plotheight))
    ax.bar(lv_labels, r2x_cumulative, label=f'Cumulative R2{yaxlbl}')
    ax.set_title(f"Cumulative R2{yaxlbl} {addtitle}")
    ax.set_xlabel('Components')
    ax.set_ylabel(f'Cumulative R2{yaxlbl}')
    plt.show()

    if is_pls:
        r2_values = list(r2Y_dict.values())
        r2y_cumulative = np.cumsum(r2_values)
        # Plot Cumulative R2Y as bar chart
        fig, ax = plt.subplots(figsize=(plotwidth, plotheight))
        ax.bar(components, r2y_cumulative, label='Cumulative R2Y')
        ax.set_title(f"Cumulative R2Y {addtitle}")
        ax.set_xlabel('Components')
        ax.set_ylabel('Cumulative R2Y')
        plt.show()

def vip(mvm_obj: Dict, *, plotwidth: int = 10, material: Union[bool, str] = False, zspace: bool = False, addtitle: str = '') -> None:
    """
    Generates a bar plot for Variable Importance in Projection (VIP) for the given multivariate model object.

    Args:
    - mvm_obj (Dict): The multivariate model object.
    - plotwidth (int, optional): The width of the plot. Defaults to 10.
    - material (Union[bool, str], optional): Material specification for the plot. Defaults to False.
    - zspace (bool, optional): Flag to indicate Z-space in the plot. Defaults to False.
    - addtitle (str, optional): Additional title for the plot. Defaults to ''.

    Returns:
        None
    """

    mvm_obj = mvm_obj.copy()
    if 'Q' in mvm_obj:  
        if (mvm_obj['type'] == 'lpls') or (mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls'):
            if (mvm_obj['type'] == 'lpls'):
                mvm_obj['Ws'] = mvm_obj['Ss']
            if isinstance(material, bool) and not(zspace):
                mvm_obj['Ws'] = mvm_obj['Ss']
            if ((mvm_obj['type'] == 'jrpls') or (mvm_obj['type'] == 'tpls')) and not(isinstance(material, bool)):
                mvm_obj['Ws'] = mvm_obj['Ssi'][mvm_obj['materials'].index(material)]
                mvm_obj['varidX'] = mvm_obj['varidXi'][mvm_obj['materials'].index(material)]
            elif (mvm_obj['type'] == 'tpls') and zspace:
                mvm_obj['varidX'] = mvm_obj['varidZ']
            
        else:
            num_varX = mvm_obj['P'].shape[0]
            # rnd_num = str(int(np.round(1000 * np.random.random_sample())))
            # rnd_num = timestr()
            # output_file("VIP_" + rnd_num + ".html", title='VIP Coefficient', mode='inline') 
                   
        if 'varidX' in mvm_obj:
            XVar = mvm_obj['varidX']
        else:
            XVar = []
            for n in list(np.arange(num_varX) + 1):
                XVar.append('XVar #' + str(n))               
        
        vip = np.sum(np.abs(mvm_obj['Ws'] * np.tile(mvm_obj['r2y'], (mvm_obj['Ws'].shape[0], 1))), axis=1)
        vip = np.reshape(vip, (len(vip), -1))
        sort_indx = np.argsort(-vip, axis=0)
        vip = vip[sort_indx]
        sorted_XVar = [XVar[i] for i in sort_indx[:, 0]]
        print(vip)
        
        # Matplotlib plotting
        plt.figure(figsize=(plotwidth, 6))
        plt.bar(sorted_XVar, vip.flatten(), color='blue')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Variables')
        plt.ylabel('Very Important to the Projection')
        plt.title('VIP ' + addtitle)
        plt.tight_layout()
        plt.show()
