from resnest import ResNeSt50

weights = 'imagenet'
include_top = True
pooling = 'avg'

model = ResNeSt50(input_shape=[224, 224, 3], weights=weights, include_top=include_top, pooling=pooling)
print('ResNeSt50(dilation=1):', model.input, model.output, sep='\n', end='\n\n')

model = ResNeSt50(input_shape=[224, 224, 3], weights=weights, include_top=include_top, pooling=pooling, dilation=2)
print('ResNeSt50(dilation=2):', model.input, model.output, sep='\n', end='\n\n')

model = ResNeSt50(input_shape=[224, 224, 3], weights=weights, include_top=include_top, pooling=pooling, dilation=4)
print('ResNeSt50(dilation=4):', model.input, model.output, sep='\n', end='\n\n')
