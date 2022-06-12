import tensorflow as tf

import resnest


class ResNeStTestMixin:
    _model_fn = None

    IN_SHAPE = [224, 224, 3]
    OUT_SHAPE = [None, 7, 7, 2048]
    OUT_D4_SHAPE = [None, 28, 28, 2048]
    OUT_D2_SHAPE = [None, 14, 14, 2048]

    def test_build_with_shape(self):
        model = self._model_fn(
            input_shape=self.IN_SHAPE,
            weights=None,
            include_top=True
        )

        self.assertEqual([None, 1000], model.output.shape.as_list())

    def test_output_shape_with_top(self):
        expected_classes = 1000
        input_tensor = tf.keras.Input(self.IN_SHAPE, name="images")
        model = self._model_fn(
            input_tensor=input_tensor,
            weights=None,
            include_top=True
        )

        self.assertEqual([None, expected_classes], model.output.shape.as_list())
        self.assertEqual(model.layers[-1].activation, tf.keras.activations.softmax)
    
    def test_output_shape_with_top_20(self):
        expected_classes = 20
        input_tensor = tf.keras.Input(self.IN_SHAPE, name="images")
        model = self._model_fn(
            input_tensor=input_tensor,
            weights=None,
            include_top=True,
            classes=expected_classes,
            classifier_activation='sigmoid',
        )

        self.assertEqual([None, expected_classes], model.output.shape.as_list())
        self.assertEqual(model.layers[-1].activation, tf.keras.activations.sigmoid)

    def test_output_shape_without_top(self):
        input_tensor = tf.keras.Input(self.IN_SHAPE, name="images")
        model = self._model_fn(
            input_tensor=input_tensor,
            weights=None,
            include_top=False,
            pooling=None,
        )

        self.assertEqual(self.OUT_SHAPE, model.output.shape.as_list())
    
    def test_dilation_2(self):
        input_tensor = tf.keras.Input(self.IN_SHAPE, name="images")
        model = self._model_fn(
            input_tensor=input_tensor,
            weights=None,
            include_top=False,
            pooling=None,
            dilation=2,
        )
        self.assertEqual(self.OUT_D2_SHAPE, model.output.shape.as_list())

    def test_dilation_4(self):
        input_tensor = tf.keras.Input(self.IN_SHAPE, name="images")
        model = self._model_fn(
            input_tensor=input_tensor,
            weights=None,
            include_top=False,
            pooling=None,
            dilation=4,
        )
        self.assertEqual(self.OUT_D4_SHAPE, model.output.shape.as_list())


class ResNeSt50Test(ResNeStTestMixin, tf.test.TestCase):
    _model_fn = staticmethod(resnest.ResNeSt50)


class ResNeSt101Test(ResNeStTestMixin, tf.test.TestCase):
    _model_fn = staticmethod(resnest.ResNeSt101)


class ResNeSt200Test(ResNeStTestMixin, tf.test.TestCase):
    _model_fn = staticmethod(resnest.ResNeSt200)


class ResNeSt269Test(ResNeStTestMixin, tf.test.TestCase):
    _model_fn = staticmethod(resnest.ResNeSt269)
