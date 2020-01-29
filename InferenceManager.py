import os
from config import get_config
from plot_qualitative_results_given_2_imgs import initialize_learner,\
                                                  plotResults,\
                                                  getPairedTensors


class InferenceManager():
    def __init__(self, conf, mdl_name, exdir, dataset_name):
        os.makedirs(exdir, exist_ok=True)
        self.conf = conf
        self.exdir = exdir
        self.dataset_name = dataset_name
        self.learner = initialize_learner(conf, mdl_name)

    def infer(self, img_name_1, img_name_2, result_filename):
        print('>>>>In inferManager, infer', img_name_1, img_name_2, result_filename)
        img_base64, meta = plotResults(self.conf, self.learner, self.exdir,
                                 img_name_1, img_name_2, result_filename,
                                 self.dataset_name)
        return img_base64, meta

    def inferWithoutPlotting(self, upload_filename, filesToCompare):
        image_stack = getPairedTensors(upload_filename, filesToCompare)
        xCoses = self.learner.getXCos(image_stack, self.conf,
                                      tta=False, attention=None)
        results = []
        for fname, xCos_value in zip(filesToCompare, xCoses):
            results.append({
                'filename': os.path.basename(fname),
                'score': float(xCos_value)
                })
        return results


if __name__ == "__main__":
    # Define the model config
    mdl_name_default = \
        '2019-08-25-14-35_accuracy:0.9931666666666666_step:218349_None.pth'
    exdir = '/home/r07944011/demo/xcos-demo/public/results'
    dataset = 'lfw'
    dataset_name = dataset
    conf = get_config(training=False)
    # Why bs_size can only be the number that divide 6000 well?
    conf.batch_size = 200
    # Initialize the model manager
    inferenceManager = InferenceManager(conf, mdl_name_default,
                                        exdir, dataset_name)
    test_upload_filename = 'livedemo_2020-1-7-0-3-8.jpeg'
    test_filesToCompare = [
        'winston_2.jpg',
        'obama_sunglasses.jpg',
        'Chen_Liang-Ji_1.jpg'
        ]
    test_pic_dir = '/home/r07944011/demo/xcos-demo/public'
    test_upload_dir = os.path.join(test_pic_dir, 'uploaded_imgs')
    test_database_dir = os.path.join(test_pic_dir, 'database')
    test_upload_filename = os.path.join(test_upload_dir, test_upload_filename)
    test_filesToCompare = [os.path.join(test_database_dir, f)
                           for f in test_filesToCompare]
    xCos = inferenceManager.inferWithoutPlotting(test_upload_filename,
                                                 test_filesToCompare)
    print(xCos)
