#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""imbin.py: Colour quantisation based on covariance eigen values

IMBIN utilises a binary tree approach to split nodes from max
of corresponding eigenvalues and store image resulting clusters.
Conditions for collecting the most significant tree node is also included.
"""
from .nodes import nodes
from numpy import (arange, zeros, uint32, int64, float32, argmin, concatenate,
                   array, argmax, unique, uint8, dot, bincount, isnan, isinf)
from sklearn.preprocessing import minmax_scale
from numpy import sum as npsum
from numpy import abs as npabs
from numpy import min as npmin
from numpy import max as npmax
from numpy import full
from numpy.linalg import eigh
from scipy.linalg.blas import scasum, ddot
from scipy.stats import entropy
from cv2 import cvtColor, COLOR_BGR2RGB, resize, INTER_CUBIC, bitwise_and
from sys import exit as sysexit

__author__ = "K. Kamzelis, G. Chliveros, and I. Tzanetatos"
__copyright__ = "Copyright 2019, WatchOver Project"
__credits__ = ["", "", "", ""]  # people who reported bug fixes
__license__ = "LGPL"
__version__ = "1.0.1"
__maintainer__ = "TBA"
__email__ = "TBA"
__status__ = "Development"  # "Prototype", "Development", or "Production"

# TODO: **kwargs for criteria
def _dotDecomp(comps, eig):

    return scasum(comps, eig)

def _dot(vector_1, vector_2):

    return ddot(vector_1, vector_2)

def _check_dtype(data, rows, cols):
    
    if data.ndim == 3 and 'int' in data.dtype.name:
        return _imgComps(data, rows, cols)
    else:
        return _components(data)

def _imgComps(img, height, width):
    '''
    Image processing function that initially converts input image to RGB,
    resizes the input image to the desired, user-defined dimentions by a
    bicubic interpolation over 4x4 pixel neighborhodd (INTER_CUBIC).
    The now resized image is normalized by means of MINMAX. Function returns
    vectorized colour channels

    The initial conversion to RGB occurs due to how OpenCv implements RGB
    colour codes (BGR to RGB).

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    height : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    # Assert input frame validity
    if img is None:
        sysexit("Frame is empty (no input frame).")
    # Convert from BGR to RGB
    img_color = cvtColor(img, COLOR_BGR2RGB)

    rows, colms = img.shape[:2]

    # Skip resize
    if width is not None or height is not None:
        # Check for resize
        if rows != width and colms != height:
            #=============================================================================#
            #    -INTER_NEAREST  - a nearest-neighbor interpolation                       #
            #    -INTER_LINEAR   - a bilinear interpolation (used by default)             #
            #    -INTER_AREA     - resampling using pixel area relation. It may be a      #
            #                      preferred method for image decimation, as it gives     #
            #                      moire’-free results. But when the image is zoomed,     #
            #                      it is similar to the INTER_NEAREST method.             #
            #    -INTER_CUBIC    - a bicubic interpolation over 4x4 pixel neighborhood    #
            #    -INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood    #
            img_color = resize(img_color, dsize=(
                width, height), interpolation=INTER_CUBIC)

    # Return output frame
    return int64(img_color.reshape(-1, 3)), rows*colms

# TODO: optional preprocessing step
def _components(data):
    
    if data.ndim == 1:
        return data, len(data)
    elif data.ndim > 2:
        rows, colms = data.shape[:2]
        return data.reshape(-1, data.shape[2]), rows*colms
    else:
        rows, colms = data.shape
        return data.reshape(-1, 1), rows*colms

def _chnlIntensity(comps):
    '''


    Parameters
    ----------
    comps : TYPE
        DESCRIPTION.
    point : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    return int64(npsum(comps, axis=0))  # Optimize, very slow

def _info_gain(parent_entropy, N1, N2, comps_len, entropy_nodes):
    '''
    

    Parameters
    ----------
    parent_entropy : TYPE
        DESCRIPTION.
    N1 : TYPE
        DESCRIPTION.
    N2 : TYPE
        DESCRIPTION.
    comps_len : TYPE
        DESCRIPTION.
    entropy_nodes : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # TODO: instead of comps_len use parent_len although same thing more or less
    return parent_entropy - ((N1/comps_len) * entropy_nodes[0] + (N2/comps_len) * entropy_nodes[1])


def _distribution_quantization(comps):
    
    # Normalize components
    return minmax_scale(comps.flatten(), feature_range=(0, 255), axis=0)

def _nodeEntropy(comps):
    '''


    Parameters
    ----------
    comps : TYPE
        DESCRIPTION.
    point : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    n_bins = unique(comps.ravel())
    
    # Check comps dtype
    if 'int' not in comps.dtype.name:
        comps = uint8(_distribution_quantization(comps))
    
    # Probability distribution of cluster - Optimize, unique very slow
    probDist = bincount(comps.ravel(), minlength=len(n_bins))

    return entropy(probDist, base=2)


def _KLTCovar(comps, chnlInt, N):
    '''


    Parameters
    ----------
    comps : TYPE
        DESCRIPTION.
    chnlInt : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    covar = zeros((comps.shape[1], comps.shape[1]), dtype=int64)
    for i in range(comps.shape[1]):
        for k in range(comps.shape[1]):
            covar[i, k] = _dot(comps[:, i], comps[:, k])  # Optimize

    return int64(covar - (_dot(chnlInt, chnlInt.T) / N))


def _eigs(kltcovar):
    '''


    Parameters
    ----------
    kltcovar : TYPE
        DESCRIPTION.

    Returns
    -------
    eigvecs : TYPE
        DESCRIPTION.
    eigvals : TYPE
        DESCRIPTION.

    '''

    eigvals, eigvecs = eigh(kltcovar)

    return eigvecs, eigvals


def _minEigvec(eigvecs):
    '''


    Parameters
    ----------
    eigvecs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    pco = npabs(npsum(eigvecs, axis=1))  # Optimize, very slow

    pco_idx = argmin(pco)

    return eigvecs[:, pco_idx]


def _minEigval(eigval):
    '''


    Parameters
    ----------
    eigval : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    return npmin(npabs(eigval))


def _quantization(M, N):
    '''


    Parameters
    ----------
    M : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    return M / N


def _eigenDecomp(imgComps, eigvec, Q, point):
    '''


    Parameters
    ----------
    imgComps : TYPE
        DESCRIPTION.
    eigvec : TYPE
        DESCRIPTION.
    point : TYPE
        DESCRIPTION.
    Q : TYPE
        DESCRIPTION.

    Returns
    -------
    point1 : TYPE
        DESCRIPTION.
    point2 : TYPE
        DESCRIPTION.

    '''
    decomp = float32(0)
    # Eigen decomposition of input node - Optimize, dot slow
    for i in range(imgComps.shape[1]):
        decomp += dot(eigvec[i], imgComps[:, i])

    return point[decomp >= dot(eigvec, Q)], point[decomp < dot(eigvec, Q)]


def _nodeForming(comps, criteria, N, parent, itr, nodeself, point=None):
    '''


    Parameters
    ----------
    comps : TYPE
        DESCRIPTION.
    criteria : string
        DESCRIPTION.
    N : int
        DESCRIPTION.
    parent : int
        DESCRIPTION.
    itr : int
        DESCRIPTION.
    nodeself : int
        DESCRIPTION.
    point : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    # Intensity of each colour channel in current node
    M = _chnlIntensity(comps)

    # Colour channel covariance
    covar = _KLTCovar(comps, M, N)

    # Eigenvectors & eigenvalues of parent node
    eigvec, eigval = _eigs(covar)

    # Instantiate node
    node = nodes(N)

    # Covariance of current node by KLT
    node.covariance = covar

    # Intensity of current node
    node.intensity = M

    # Point of current node
    node.point = point

    # Eigenvector of current node
    node.eigvec = _minEigvec(eigvec)

    # Eigenvalue of current node
    node.eigvals = _minEigval(eigval)

    # Entropy of current node
    node.entropy = _nodeEntropy(comps)

    # Parent node of current node
    node.parent = uint8(parent)

    # Training iteration
    node.itr = uint8(itr)

    # Node self
    node.nodeself = uint8(nodeself)

    # Return entropy split criteria
    if criteria == 'entropy':
        return node, node.entropy

    # Return eigenvalue split criteria - no reason to return eigvals since I'm returning the node
    else:
        return node, node.eigvals

def _Decompose(K, comps, N, criteria='eigen'):
    '''


    Parameters
    ----------
    K : TYPE
        DESCRIPTION.
    comps : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    criteria : TYPE, optional
        DESCRIPTION. The default is 'eigen'.

    Returns
    -------
    pNodes : TYPE
        DESCRIPTION.
    leafNodes : TYPE
        DESCRIPTION.

    '''

    leafNodes = zeros((K, 1), dtype=object)
    splitcrit = zeros((K, 1), dtype=float32)
    pNodes = zeros((K-2, 1), dtype=object)
    treeState = full(K, fill_value=None, dtype=object)

    leafNodes[0], splitcrit[0] = _nodeForming(comps=comps,
                                              criteria=criteria,
                                              N=N,
                                              parent=0,
                                              itr=0,
                                              nodeself=0,
                                              point=arange(0, N, dtype=uint32))
    
    treeState [0] = leafNodes[0]

    i = 1
    while i <= K-1:

        if npmax(splitcrit == -1):
            break
        idx = argmax(splitcrit)

        Q = _quantization(leafNodes[idx][0].intensity,
                          leafNodes[idx][0].cluster_size)

        point1, point2 = _eigenDecomp(comps[leafNodes[idx][0].point, :],
                                      leafNodes[idx][0].eigvec,
                                      Q,
                                      point=leafNodes[idx][0].point)

        N1 = uint32(len(point1))
        N2 = uint32(len(point2))

        if N1 > 0 and N2 > 0:

            if i > 1:
                leafNodes[idx][0].children = array((idx, i), dtype=uint8)
                leafNodes[idx][0].childIter = uint8(i-1)

                pNodes[i-2] = leafNodes[idx]

            leafNodes[idx], splitcrit[idx] = _nodeForming(comps=comps[point1, :],
                                                          criteria=criteria,
                                                          N=N1,
                                                          parent=idx,
                                                          itr=i-1,
                                                          nodeself=idx,
                                                          point=point1)

            leafNodes[i], splitcrit[i] = _nodeForming(comps=comps[point2, :],
                                                      criteria=criteria,
                                                      N=N2,
                                                      parent=idx,
                                                      itr=i-1,
                                                      nodeself=i,
                                                      point=point2)

            treeState[i-1] = leafNodes[:i+1]

            i += 1
        else:
            splitcrit[idx] = -1
            i -= 1
    return pNodes, leafNodes, treeState


def _greedy_decompose(K, comps, N, criteria='eigen'):
    '''
    

    Parameters
    ----------
    comps : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    criteria : TYPE, optional
        DESCRIPTION. The default is 'eigen'.

    Returns
    -------
    pNodes : TYPE
        DESCRIPTION.
    leafNodes : TYPE
        DESCRIPTION.
    treeState : TYPE
        DESCRIPTION.

    '''
    leafNodes = zeros((K, 1), dtype=object)
    splitcrit = zeros((K, 1), dtype=float32)
    entropy_tree = zeros((K, 1), dtype=float32)
    info_gain = zeros((K, 1), dtype=float32)
    pNodes = zeros((K-2, 1), dtype=object)
    treeState = full(K, fill_value=None, dtype=object)

    leafNodes[0], splitcrit[0] = _nodeForming(comps=comps,
                                              criteria=criteria,
                                              N=N,
                                              parent=0,
                                              itr=0,
                                              nodeself=0,
                                              point=arange(0, N, dtype=uint32))

    entropy_tree[0] = leafNodes[0][0].entropy

    i = 1
    while i <= K-1:

        if npmax(splitcrit == -1):
            break
        idx = argmax(splitcrit)

        parent_entropy = entropy_tree[idx]

        Q = _quantization(leafNodes[idx][0].intensity,
                          leafNodes[idx][0].cluster_size)

        point1, point2 = _eigenDecomp(comps[leafNodes[idx][0].point, :],
                                      leafNodes[idx][0].eigvec,
                                      Q,
                                      point=leafNodes[idx][0].point,
                                      )

        N1 = uint32(len(point1))
        N2 = uint32(len(point2))

        if N1 > 0 and N2 > 0:

            if i > 1:
                leafNodes[idx][0].children = array((idx, i), dtype=uint8)
                leafNodes[idx][0].childIter = uint8(i-1)

                pNodes[i-2] = leafNodes[idx]

            leafNodes[idx], splitcrit[idx] = _nodeForming(comps=comps[point1, :],
                                                          criteria=criteria,
                                                          N=N1,
                                                          parent=idx,
                                                          itr=i-1,
                                                          nodeself=idx,
                                                          point=point1)

            leafNodes[i], splitcrit[i] = _nodeForming(comps=comps[point2, :],
                                                      criteria=criteria,
                                                      N=N2,
                                                      parent=idx,
                                                      itr=i-1,
                                                      nodeself=i,
                                                      point=point2)
            entropy_tree[idx] = leafNodes[idx][0].entropy
            entropy_tree[i] = leafNodes[i][0].entropy


            if i > 1:
                info_gain[i-1] = _info_gain(parent_entropy,
                                          N1,
                                          N2,
                                          comps_len=pNodes[i-2][0].cluster_size,
                                          entropy_nodes=[entropy_tree[idx], entropy_tree[i]],
                                          )
                # TODO: Estimate info gain based on previous splits (?)
                if info_gain[i-2] > info_gain[i-1]:
                    break


            treeState[i - 1] = leafNodes[:i + 1]

            i += 1
        else:
            splitcrit[idx] = -1
            i -= 1
    return treeState

def _check_valid_states(tree_state):
    
    valid_trees = []
    for tree in tree_state:
        if tree is not None:
            valid_trees.append(tree)
    return valid_trees

def _fastEigenPruning(tree_state):

    valid_trees = _check_valid_states(tree_state)
    
    if len(valid_trees) == 1:
        return tree_state[0]
    
    depthEntropy = zeros(len(valid_trees), dtype=float32)

    for i, tree in enumerate(valid_trees):
        for node in tree:
            depthEntropy[i] += node[0].entropy

    index = argmax(depthEntropy)

    return valid_trees[index]


def _EigenPruning(pNodes, leafNodes):
    '''


    Parameters
    ----------
    pNodes : TYPE
        DESCRIPTION.
    leafNodes : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    # @cuda.jit(device=True)
    def _pruneDec(pNodes, leafNodes):
        '''


        Parameters
        ----------
        pNodes : TYPE
            DESCRIPTION.
        leafNodes : TYPE
            DESCRIPTION.
        children : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        '''

        nodeEntropy = zeros((len(leafNodes)+1, 1), dtype=float32)
        nodeEntropy[0] = pNodes.entropy

        for i in range(1, len(leafNodes)+1):
            if leafNodes[i-1][0] is not None:
                nodeEntropy[i] = leafNodes[i-1][0].entropy

        if argmax(nodeEntropy) == 0:
            return True
        else:
            return False

    for i in range(len(pNodes)-1, -1, -1):

        cnt = 0
        children = zeros(len(pNodes[i][0].children), dtype=uint8)

        for k in range(len(leafNodes)):
            if leafNodes[k][0] is not None:
                for j in range(len(children)):
                    # Locate child nodes of current parent node
                    if pNodes[i][0].childIter == leafNodes[k][0].itr and \
                            pNodes[i][0].nodeself == leafNodes[k][0].parent and \
                            pNodes[i][0].children[j] == leafNodes[k][0].nodeself:

                        children[cnt] = k
                        cnt += 1
        if _pruneDec(pNodes[i][0], leafNodes[children]):
            # Remove right child & restore parent node
            for k in range(len(children)):
                if leafNodes[children[k]][0] is not None:
                    # Check if child or grandchild ?
                    if pNodes[i][0].nodeself == leafNodes[children[k]][0].nodeself and \
                            pNodes[i][0].childIter == leafNodes[children[k]][0].itr:
                        leafNodes[children[k]] = pNodes[i]
                    else:
                        leafNodes[children[k]] = None
        else:
            if i > 0:
                for k in range(i-1, -1, -1):
                    if pNodes[k][0].childIter == pNodes[i][0].itr and \
                        pNodes[k][0].nodeself == pNodes[i][0].parent and \
                        (pNodes[k][0].children[0] == pNodes[i][0].nodeself or
                         pNodes[k][0].children[1] == pNodes[i][0].nodeself):

                        # TODO: Preserve children relationships (do not believe grandchildren are children)
                        pNodes[k][0].children = unique(
                            concatenate((pNodes[k][0].children, children)))
                # TODO: Preserve children properties (do not confirm grandparent's belief)
                for k in range(len(children)):
                    leafNodes[children[k]][0].itr = pNodes[i][0].itr
                    leafNodes[children[k]][0].parent = pNodes[i][0].parent

    return leafNodes


def imbin(img, K, rows=None, cols=None, criteria='eigen', pruned=True):
    '''


    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    rows : TYPE, optional
        DESCRIPTION. The default is None.
    cols : TYPE, optional
        DESCRIPTION. The default is None.
    criteria : TYPE, optional
        DESCRIPTION. The default is 'eigen'.
    pruned : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    comps, N = _check_dtype(img, rows, cols)

    pnodes, leafnodes, tree_state = _Decompose(K, comps, N, criteria=criteria)

    if pruned:
        # leafnodes = _EigenPruning(pnodes, leafnodes)
        return _fastEigenPruning(tree_state)
    else:
        return leafnodes


def imbin_predict(img, K, rows=None, cols=None, criteria='eigen', pruned=True, return_frame=True):
    '''


    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    rows : TYPE, optional
        DESCRIPTION. The default is None.
    cols : TYPE, optional
        DESCRIPTION. The default is None.
    criteria : TYPE, optional
        DESCRIPTION. The default is 'eigen'.
    pruned : TYPE, optional
        DESCRIPTION. The default is True.
    return_frame : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if isnan(img).any():
        # Set invalid elements to zero
        img = isnan(img, uint8(0))
    if isinf(img).any():
        img = isinf(img, uint8(255))
    tree = imbin(img, K, rows, cols, criteria, pruned)

    return predict(tree, return_frame, img)


def greedy_imbin(img, K=10, rows=None, cols=None, criteria='eigen'):
    '''
    

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    rows : TYPE, optional
        DESCRIPTION. The default is None.
    cols : TYPE, optional
        DESCRIPTION. The default is None.
    criteria : TYPE, optional
        DESCRIPTION. The default is 'eigen'.

    Returns
    -------
    tree : TYPE
        DESCRIPTION.

    '''
    
    comps, N = _check_dtype(img, rows, cols)
    
    tree = _greedy_decompose(K=K,
                             comps=comps,
                             N=N,
                             criteria='eigen',
                             )
    
    return tree


def greedy_imbin_predict(img, K=10, rows=None, cols=None, criteria='eigen',
                               return_frame=True):
    '''
    

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    rows : TYPE, optional
        DESCRIPTION. The default is None.
    cols : TYPE, optional
        DESCRIPTION. The default is None.
    criteria : TYPE, optional
        DESCRIPTION. The default is 'eigen'.
    return_frame : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    tree_state = greedy_imbin(img, K, rows, cols, criteria)

    tree = _fastEigenPruning(tree_state)

    return predict(tree, return_frame, img)


def predict(leafnodes, return_frame=True, img=None):
    '''


    Parameters
    ----------
    leafnodes : TYPE
        DESCRIPTION.
    return_frame : TYPE
        DESCRIPTION.
    img : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    valid_leafnodes = _check_valid_states(leafnodes)
    if len(valid_leafnodes) == 1:
        if return_frame:
            return img
        else:
            return valid_leafnodes[0]

    leafEntropy = zeros(len(leafnodes), dtype=float32)

    for i, leafnode in enumerate(leafnodes):
        if leafnode[0] is not None:
            leafEntropy[i] = leafnode[0].entropy

    if return_frame:

        assert img is not None, "No reference frame entered."

        return reconstruct(leafnodes[argmax(leafEntropy)], img)

    else:

        return leafnodes[argmax(leafEntropy)][0]


def reconstruct(prednode, img):
    '''


    Parameters
    ----------
    prednode : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    pred_img : TYPE
        DESCRIPTION.

    '''

    pred_mask = zeros((img.shape[0]*img.shape[1]), dtype=uint8)

    pred_mask[prednode[0].point] = uint8(255)

    pred_mask = pred_mask.reshape(img.shape[:2], order='C')

    return bitwise_and(img, img, mask=pred_mask)
