"""
Incomplete Code
dataset builder for Pascal VOC segmentation
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import os

class voc2007Semantic(tfds.image.Voc2007):
    """ Pascal VOC 2007 - Semantic Segmentation"""
    VERSION = tfds.core.Version("0.1.0")
    def _info(self):
        # specifies the tfds.core.DatasetInfo object
        parent_info = tfds.image.Voc2007().info
        return tfds.core.DatasetInfo(
            builder=self,
            description=parent_info.description,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "image/filename": tfds.features.Text(),
                    "label": tfds.features.Image(shape=(None, None, 1))
                }
            ),
            urls=parent_info.urls,
            citation=parent_info.citation,
        )

    def _split_generators(self, dl_manager):
        trainval_path = dl_manager.download_and_extract(
            os.path.join("/home/swagat/tensorflow_datasets/downloads",
            "pjreddi.com_media_files_VOCtrai_6-Nov-2007fYzZURAbCVfd_XpTC9yKlPBhIc_B5RG7WTfpcwIMdQg.tar")
         )
        test_path = dl_manager.download_and_extract(
            os.path.join("/home/swagat/tensorflow_datasets/downloads",
            "pjreddi.com_media_files_VOCtest_6-Nov-2007aDaIji4B3KhFd6hJ0zn6T3Ph5PE10xJDDEhWtWCbSJI.tar")
            #os.path.join(_VOC2007_DATA_URL, "VOCtest_06-Nov-2007.tar")
        )
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=1,
                gen_kwargs=dict(data_path=test_path, set_name="test")),
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=1,
                gen_kwargs=dict(data_path=trainval_path, set_name="train")),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                num_shards=1,
                gen_kwargs=dict(data_path=trainval_path, set_name="val"))
            ]

    # def _split_generators(self, dl_manager):
    #     # downloads the data and defines the split
    #     # dl_manager is a tfds.download.DownloadManager
    #     pass


    def _generate_examples(self, data_path, set_name):
        set_filepath = os.path.join(
            data_path,
            "VOCdevkit/VOC2007/ImageSets/Segmentation/{}.txt".format(set_name),
        )
        with tf.io.gfile.GFile(set_filepath, "r") as f:
            for line in f:
                image_id = line.strip()

                image_filepath = os.path.join(
                    data_path, "VOCdevkit", "VOC2007", "JPEGImages", f"{image_id}.jpg"
                )
                label_filepath = os.path.join(
                    data_path,
                    "VOCdevkit",
                    "VOC2007",
                    "SegmentationClass",
                    f"{image_id}.png",
                )

                if not tf.io.gfile.exists(label_filepath):
                    continue

                label_rgb = tf.image.decode_image(
                    tf.io.read_file(label_filepath), channels=3
                )

                label = tf.Variable(
                    tf.expand_dims(
                        tf.zeros(shape=tf.shape(label_rgb)[:-1], dtype=tf.uint8), -1
                    )
                )

                for color, label_id in LUT.items():
                    match = tf.reduce_all(tf.equal(label_rgb, color), axis=[2])
                    labeled = tf.expand_dims(tf.cast(match, tf.uint8), axis=-1)
                    label.assign_add(labeled * label_id)

                colored = tf.not_equal(tf.reduce_sum(label), tf.constant(0, tf.uint8))
                # Certain labels have wrong RGB values
                if not colored.numpy():
                    tf.print("error parsing: ", label_filepath)
                    continue

                yield image_id, {
                    # Declaring in _info "image" as a tfds.feature.Image
                    # we can use both an image or a string. If a string is detected
                    # it is supposed to be the image path and tfds take care of the
                    # reading process.
                    "image": image_filepath,
                    "image/filename": f"{image_id}.jpg",
                    "label": label.numpy(),
                }

dataset, info = tfds.load("voc2007_semantic", with_info=True)
print(info)

