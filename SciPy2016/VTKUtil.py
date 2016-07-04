# paperVTKUtil
# Functions used to make figure for the papers.
import numpy as np, sys, vtk, os, vtk.util.numpy_support as npsup
import time, matplotlib as mpl, gc
# from scipy.stats._support import unique as uniqueRows

def addAxes(screenSize,ren,xRange,yRange,zRange,axesBounds):
    axes2 = vtk.vtkCubeAxesActor()
    axes2.SetCamera(ren.GetActiveCamera())
    axes2.SetScreenSize(screenSize)
    axes2.SetFlyModeToOuterEdges()
    axes2.SetEnableDistanceLOD(0)
    axes2.SetEnableViewAngleLOD(0)
    axes2.SetLabelScaling(0,0,0,0)


    axes2.SetXLabelFormat("%3.0f")
    axes2.SetXTitle('Easting [UTM  km]')
    axes2.SetXAxisRange(xRange)
    axes2.GetTitleTextProperty(0).SetColor(0.0,0.0,0.0)
    axes2.GetLabelTextProperty(0).SetColor(0.0,0.0,0.0)
    axes2.GetLabelTextProperty(0).SetOrientation(0.0)
    axes2.GetLabelTextProperty(0).SetFontSize(50)
    axes2.GetLabelTextProperty(0).SetVerticalJustificationToCentered()
    axes2.GetXAxesGridlinesProperty().SetColor(0.0,0.0,0.0)
    axes2.GetXAxesLinesProperty().SetColor(0.0,0.0,0.0)
    axes2.DrawXGridlinesOff()
    axes2.DrawXInnerGridlinesOff()

    axes2.SetYLabelFormat("%3.0f")
    axes2.SetYTitle('Northing [UTM  km]')
    axes2.SetYAxisRange(yRange)
    axes2.GetTitleTextProperty(1).SetColor(0.0,0.0,0.0)
    axes2.GetLabelTextProperty(1).SetColor(0.0,0.0,0.0)
    axes2.GetLabelTextProperty(1).SetOrientation(0.0)
    axes2.GetYAxesGridlinesProperty().SetColor(0.0,0.0,0.0)
    axes2.GetYAxesLinesProperty().SetColor(0.0,0.0,0.0)
    axes2.DrawYGridlinesOff()
    axes2.DrawYInnerGridlinesOff()

    axes2.SetZLabelFormat('%.0f')
    axes2.SetZTitle('Depth [km  b.s.l.]')
    axes2.SetZAxisRange(zRange)
    axes2.GetTitleTextProperty(2).SetColor(0.0,0.0,0.0)
    axes2.GetLabelTextProperty(2).SetColor(0.0,0.0,0.0)
    axes2.GetLabelTextProperty(2).SetOrientation(0.0)
    axes2.GetZAxesGridlinesProperty().SetColor(0.0,0.0,0.0)
    axes2.GetZAxesLinesProperty().SetColor(0.0,0.0,0.0)
    axes2.DrawZGridlinesOff()
    axes2.DrawZInnerGridlinesOff()
    axes2.SetBounds(axesBounds)
    return axes2

def makeLookupTable(lutName):

    if lutName == 'Res':
        # Paraview color table, manually exported in the paraview console
        ctfValueTable = [[1.0, 1.0, 0.0, 0.0],
        [ 10.000, 1.0, 0.7529411764705882, 0.0],
        [ 25.8867, 0.8549019607843137, 1.0, 0.0],
        [ 68.3938, 0.0, 1.0, 0.3333333333333333],
        [ 122.708, 0.0, 0.8470588235294118, 1.0],
        [ 304.544, 0.0, 0.34509803921568627, 1.0],
        [ 1000, 0.0, 0.0, 1.0]]
        colRange = [1.0, 1000.0]
        sbTitle = 'Ohm*m'
        useLog = 1
        vecName = 'Ohm*m'
        scalFor = '%-#4.0f'
        sBarnrLabels = 4
    elif lutName == 'Con':
        # Paraview color table, manually exported in the paraview console
        ctfValueTable = [
        [ 0.00001    , 0.0, 0.0, 1.0],
        [ 0.00003283 , 0.0, 0.34509803921568627, 1.0],
        [ 0.000081494, 0.0, 0.8470588235294118, 1.0],
        [ 0.000146212, 0.0, 1.0, 0.3333333333333333],
        [ 0.000386298, 0.8549019607843137, 1.0, 0.0],
        [ 0.001      , 1.0, 0.7529411764705882, 0.0],
        [ 0.01       , 1.0, 0.0, 0.0]]
        colRange = [0.00001, 0.01]
        sbTitle = 'S/m'
        useLog = 1
        vecName = 'S/m'
        scalFor = '%-#1.2e'
        sBarnrLabels = 4
    elif lutName == 'wellTemp':
        ctfValueTable = [[0,0.666667,0,1],
        [1,0,0.0313725,1],
        [100,0,0.254902,1],
        [150,0,0.921569,1],
        [200,0,1,0.545098],
        [230,0.870588,1,0],
        [260,1,0.423529,0],
        [275,1,0.144259,0],
        [350,1,0,0]]
        # <NaN r="0.486275" g="0.486275" b="0.486275"/>
        colRange = [0, 350]
        sbTitle = 'Temp [C]'
        useLog = 0
        vecName = 'WellTemp'
        scalFor = '%-#4.0f'
        sBarnrLabels = 8
    ctf = vtk.vtkColorTransferFunction()
    ctf.Build()
    ctf.SetVectorModeToMagnitude()
    ctf.SetScale(useLog)
    for nr,arr in enumerate(ctfValueTable):
        ctf.AddRGBPoint(arr[0],arr[1],arr[2],arr[3])

    ctf.AllowDuplicateScalarsOn()

    # Build a lookuptable on populate it with values from the ctf. To this the get
    # the scalar bar to work properly
    if useLog:
        lut = vtk.vtkLookupTable()
        lut.Build()
        lut.SetVectorModeToMagnitude()
        lut.SetRange(colRange)
        lut.SetTable(ctf.MapScalars(npsup.numpy_to_vtk(np.logspace(np.log10(colRange[0]),np.log10(colRange[1]),2000)),0,0))
        lut.SetScaleToLog10()
        lut.SetNanColor(0.5,0.5,0.5,0.5)
    else:
        lut = ctf
        ctf.SetNanColor(0.5,0.5,0.5)

    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(lut)
    scalarBar.SetTitle(sbTitle)
    scalarBar.SetDrawBackground(0)
    scalarBar.GetTitleTextProperty().SetColor(0,0,0)
    scalarBar.GetTitleTextProperty().SetFontSize(30)
    scalarBar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    scalarBar.GetPositionCoordinate().SetValue(0.01,0.25)
    scalarBar.GetLabelTextProperty().SetColor(0,0,0)
    scalarBar.GetLabelTextProperty().SetFontSize(30)
    scalarBar.GetFrameProperty().SetOpacity(0.0)
    scalarBar.SetNumberOfLabels(sBarnrLabels)
    scalarBar.SetLabelFormat(scalFor)
    scalarBar.SetOrientationToVertical()
    scalarBar.SetWidth(0.1)
    scalarBar.SetHeight(0.7)
    scalarBar.SetDrawFrame(01)
    return lut, scalarBar, vecName


## Add the text labels
def makeTextActor(text,size,pos):
    at = vtk.vtkVectorText()
    at.SetText(text)
    tMap = vtk.vtkPolyDataMapper()
    tMap.SetInputConnection(at.GetOutputPort())
    tAct = vtk.vtkFollower()
    tAct.SetMapper(tMap)
    tAct.GetProperty().SetColor(0,0,0)
    tAct.SetScale(size)
    tAct.SetPosition(pos)
    tAct.PickableOff()
    return tAct

def addDirectionWidget(iren,ren,coneH,coneR):
    ''' Function to add a direction widget.
    Inputs:
        iren - interactive renderer
        coneH - height of the cone
        coneR - radius of the cone

    '''
    # Define the direction cone
    nArr = vtk.vtkConeSource()
    nArr.SetResolution(100)
    nArr.SetHeight(coneH)
    nArr.SetRadius(coneR)
    nArr.SetDirection(.0,1.0,.0)
    nArr.Update()
    # Cone mapper and actor
    nArrMap = vtk.vtkPolyDataMapper()
    nArrMap.SetInputConnection(nArr.GetOutputPort())
    nArrAct = vtk.vtkActor()
    nArrAct.SetMapper(nArrMap)
    # Orientation widget
    oriWid = vtk.vtkOrientationMarkerWidget()
    oriWid.SetOrientationMarker(nArrAct)
    oriWid.SetInteractor(iren)
    oriWid.SetViewport(.05,.05,.15,.15)
    oriWid.EnabledOn()
    oriWid.InteractiveOn()
    # Add the text actor.
    oriText = vtk.vtkTextActor()
    oriText.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    oriText.GetTextProperty().SetFontSize(32)
    oriText.GetTextProperty().SetColor(0.0,0.0,0.0)
    oriText.SetPosition(0.025,0.025)
    oriText.SetInput('North')

    ren.AddActor(oriText)
    return oriWid

def makeTextCaption(text,size,labPos,attPos):
    coneGlyph = vtk.vtkConeSource()
    coneGlyph.SetResolution(10)
    coneGlyph.SetHeight(150)
    coneGlyph.SetRadius(50)
    coneGlyph.Update()
    glyphMaxSize = 50
    glyphSize = .1

    capt = vtk.vtkCaptionActor2D()
    capt.BorderOff()
    capt.SetCaption(text)
    capt.GetCaptionTextProperty().SetFontSize(size)
    capt.GetCaptionTextProperty().BoldOn()
    capt.GetCaptionTextProperty().ItalicOff()
    capt.GetCaptionTextProperty().ShadowOff()
    capt.SetAttachmentPoint(attPos)
    capt.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    capt.GetPositionCoordinate().SetReferenceCoordinate(None)
    capt.GetPositionCoordinate().SetValue(labPos)
    capt.SetWidth(0.013*(len(text)+1))
    capt.SetHeight(0.1)
    capt.ThreeDimensionalLeaderOff()
    capt.SetLeaderGlyph(coneGlyph.GetOutput())
    capt.SetMaximumLeaderGlyphSize(glyphMaxSize)
    capt.SetLeaderGlyphSize(glyphSize)
    capt.GetProperty().SetColor(0,0,0)
    capt.GetProperty().SetLineWidth(5)

    return capt
