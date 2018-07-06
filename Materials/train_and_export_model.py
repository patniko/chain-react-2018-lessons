import http.client, urllib.request, urllib.parse, urllib.error, base64, os, time, json, inspect, zipfile, sys
from azure.cognitiveservices.vision.customvision.training import training_api
from msrest.exceptions import HttpOperationError
from datetime import datetime


def trainModel(api, projectId):
    try:
        print('training...')
        iteration = api.train_project(projectId)
        while (iteration.status != 'Completed'):
            print('    training status: ' + iteration.status)
            time.sleep(5)
            iteration = api.get_iteration(projectId, iteration.id)
        api.update_iteration(projectId, iteration.id, is_default=True)
        print('done training, iteration id: ' + iteration.id)
        return iteration.id
    except HttpOperationError as error:
        response = json.loads(error.response.text)
        if (response['code'] == 'BadRequestTrainingNotNeeded'):
            print(response['message']) # Nothing changed since last training
            return getLatestIterationId(api, projectId)
        else:
            print('failed to train model: {0}, response: {1}'.format(error.message, error.response.text))
            raise


def exportIteration(api, projectId, iterationId):
    try:
        print('exporting iteration...')
        api.export_iteration(projectId, iterationId, 'tensorflow')
    except HttpOperationError as error:
        response = json.loads(error.response.text)
        if (response['code'] == 'BadRequestExportAlreadyInProgress'):
            print(response['message']) # <GUID> is already queued for export
        else:
            print('failed to export iteration: {0}, response: {1}'.format(error.message, error.response.text))
            raise
    export = getLatestExport(api, projectId, iterationId)
    if (export == None):
        raise Exception('failed to export iteration, no exports found')
    while (export.status != 'Done'):
        print('    export status: ' + export.status)
        if (export.status == 'Failed'):
            raise Exception('failed to export iteration, all exports are failed')
        time.sleep(5)
        export = getLatestExport(api, projectId, iterationId)
    print('done exporting iteration, download uri: ' + export.download_uri)
    return export.download_uri


def downloadExportedModel(downloadUri, assetsRelativePath):
    print('downloading exported model...')
    downloadedFile, headers = urllib.request.urlretrieve(downloadUri)
    print('downloaded file: ' + downloadedFile)
    zipFileRef = zipfile.ZipFile(downloadedFile, 'r')
    extractFolder = os.path.join(getScriptPath(), assetsRelativePath)
    print('extracting downloaded zip to folder: ' + extractFolder)
    zipFileRef.extractall(extractFolder)
    zipFileRef.close()


def getLatestIterationId(api, projectId):
    latestId, latestDate = '', datetime.min
    print('iterations:')
    for iteration in api.get_iterations(projectId):
        print('    id: {0}, status: {1}, created: {2}, last_modified: {3}, trained_at: {4}, exportable: {5}, domain_id: {6}'.format(iteration.id, iteration.status, iteration.created, iteration.last_modified, iteration.trained_at, iteration.exportable, iteration.domain_id))
        if (iteration.status == 'Completed' and iteration.exportable and iteration.trained_at > latestDate):
            latestId = iteration.id
            latestDate = iteration.trained_at
    print('latest exportable iteration: {0}, trained at: {1}'.format(latestId, latestDate))
    return latestId


def getLatestExport(api, projectId, iterationId):
    lastExport, lastExportStatusId = None, 0
    print('exports:')
    for export in api.get_exports(projectId, iterationId):
        print('    platform: {0}, status: {1}, flavor: {2}, download_uri: {3}'.format(export.platform, export.status, export.flavor, export.download_uri))
        if (export.platform == 'TensorFlow' and getExportStatusId(export.status) > lastExportStatusId):
            lastExport = export
            lastExportStatusId = getExportStatusId(export.status)
    return lastExport


def printFields(obj):
    for prop, val in inspect.getmembers(obj):
        print('{0} = {1}'.format(prop, val))


def getScriptPath():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def getExportStatusId(status):
    if (status == 'Failed'):
        return 1
    elif (status == 'Done'):
        return 2
    elif (status == 'Exporting'):
        return 3
    else:
        return 0


if __name__ == '__main__':
    trainingKey = os.getenv('CUSTOM_VISION_SECRET')
    projectId = os.getenv('CUSTOM_VISION_PROJECT_ID')
    assetsRelativePath = os.getenv('ASSETS_RELATIVE_PATH')

    if (trainingKey == None):
        if (len(sys.argv) > 1):
            trainingKey = sys.argv[1]
        else:
            raise Exception('Custom Vision training secret not found in CUSTOM_VISION_SECRET environment variable nor in first command line argument')
    if (projectId == None):
        raise Exception('CUSTOM_VISION_PROJECT_ID environment variable not found')
    if (assetsRelativePath == None):
        raise Exception('ASSETS_RELATIVE_PATH environment variable not found')

    api = training_api.TrainingApi(trainingKey)
    iterationId = trainModel(api, projectId)
    downloadUri = exportIteration(api, projectId, iterationId)
    downloadExportedModel(downloadUri, assetsRelativePath)