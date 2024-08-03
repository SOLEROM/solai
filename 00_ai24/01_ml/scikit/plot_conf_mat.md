# PlotConfusionMatrix

```

def PlotConfusionMatrix(vY: np.ndarray, vYPred: np.ndarray, normMethod: str = None, hA: Optional[plt.Axes] = None, 
                        lLabels: Optional[List] = None, dScore: Optional[Dict] = None, titleStr: str = 'Confusion Matrix', 
                        xLabelRot: Optional[int] = None, valFormat: Optional[str] = None) -> Tuple[plt.Axes, np.ndarray]:

    # Calculation of Confusion Matrix
    mConfMat = confusion_matrix(vY, vYPred, normalize = normMethod)
    oConfMat = ConfusionMatrixDisplay(mConfMat, display_labels = lLabels)
    oConfMat = oConfMat.plot(ax = hA, values_format = valFormat)
    hA = oConfMat.ax_
    if dScore is not None:
        titleStr += ':'
        for scoreName, scoreVal in  dScore.items():
            titleStr += f' {scoreName} = {scoreVal:0.2},'
        titleStr = titleStr[:-1]
    hA.set_title(titleStr)
    hA.grid(False)
    if xLabelRot is not None:
        for xLabel in hA.get_xticklabels():
            xLabel.set_rotation(xLabelRot)

    return hA, mConfMat


```