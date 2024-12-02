��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02unknown8ɟ
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	�*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	�*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_nameconv2d_transpose_3/kernel
�
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
:@@*
dtype0
�
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:@*
dtype0
�
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_4/kernel
�
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
: @*
dtype0
�
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
: *
dtype0
�
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_5/kernel
�
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
: *
dtype0
�
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�:
value�:B�: B�9
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
�
	layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
regularization_losses
trainable_variables
	variables
	keras_api
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
regularization_losses
trainable_variables
	variables
	keras_api
 
�
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
�
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017
�
1layer_regularization_losses
regularization_losses
trainable_variables
2metrics
3layer_metrics
	variables

4layers
5non_trainable_variables
 
 
h

kernel
 bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
h

!kernel
"bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
R
>regularization_losses
?trainable_variables
@	variables
A	keras_api
h

#kernel
$bias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
h

%kernel
&bias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
h

'kernel
(bias
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
R
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
 
F
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
F
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
�
Rlayer_regularization_losses
regularization_losses
trainable_variables
Smetrics
Tlayer_metrics
	variables

Ulayers
Vnon_trainable_variables
 
h

)kernel
*bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
R
[regularization_losses
\trainable_variables
]	variables
^	keras_api
h

+kernel
,bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
h

-kernel
.bias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
h

/kernel
0bias
gregularization_losses
htrainable_variables
i	variables
j	keras_api
 
8
)0
*1
+2
,3
-4
.5
/6
07
8
)0
*1
+2
,3
-4
.5
/6
07
�
klayer_regularization_losses
regularization_losses
trainable_variables
lmetrics
mlayer_metrics
	variables

nlayers
onon_trainable_variables
US
VARIABLE_VALUEconv2d_2/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_2/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_3/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_3/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_4/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_4/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_5/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_5/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_6/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_6/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_7/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_7/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_3/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_3/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_4/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_4/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_5/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_5/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
 
 

0
 1

0
 1
�
player_regularization_losses
6regularization_losses
qlayer_metrics
rmetrics
7trainable_variables
8	variables

slayers
tnon_trainable_variables
 

!0
"1

!0
"1
�
ulayer_regularization_losses
:regularization_losses
vlayer_metrics
wmetrics
;trainable_variables
<	variables

xlayers
ynon_trainable_variables
 
 
 
�
zlayer_regularization_losses
>regularization_losses
{layer_metrics
|metrics
?trainable_variables
@	variables

}layers
~non_trainable_variables
 

#0
$1

#0
$1
�
layer_regularization_losses
Bregularization_losses
�layer_metrics
�metrics
Ctrainable_variables
D	variables
�layers
�non_trainable_variables
 

%0
&1

%0
&1
�
 �layer_regularization_losses
Fregularization_losses
�layer_metrics
�metrics
Gtrainable_variables
H	variables
�layers
�non_trainable_variables
 

'0
(1

'0
(1
�
 �layer_regularization_losses
Jregularization_losses
�layer_metrics
�metrics
Ktrainable_variables
L	variables
�layers
�non_trainable_variables
 
 
 
�
 �layer_regularization_losses
Nregularization_losses
�layer_metrics
�metrics
Otrainable_variables
P	variables
�layers
�non_trainable_variables
 
 
 
8
	0

1
2
3
4
5
6
7
 
 

)0
*1

)0
*1
�
 �layer_regularization_losses
Wregularization_losses
�layer_metrics
�metrics
Xtrainable_variables
Y	variables
�layers
�non_trainable_variables
 
 
 
�
 �layer_regularization_losses
[regularization_losses
�layer_metrics
�metrics
\trainable_variables
]	variables
�layers
�non_trainable_variables
 

+0
,1

+0
,1
�
 �layer_regularization_losses
_regularization_losses
�layer_metrics
�metrics
`trainable_variables
a	variables
�layers
�non_trainable_variables
 

-0
.1

-0
.1
�
 �layer_regularization_losses
cregularization_losses
�layer_metrics
�metrics
dtrainable_variables
e	variables
�layers
�non_trainable_variables
 

/0
01

/0
01
�
 �layer_regularization_losses
gregularization_losses
�layer_metrics
�metrics
htrainable_variables
i	variables
�layers
�non_trainable_variables
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_vae_inputPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_vae_inputconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_156357
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_157584
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_157648��
�
�
$__inference_vae_layer_call_fn_156226
	vae_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	�

unknown_10:	�$

unknown_11:@@

unknown_12:@$

unknown_13: @

unknown_14: $

unknown_15: 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	vae_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vae_layer_call_and_return_conditional_losses_1561462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:���������
#
_user_specified_name	vae_input
�
�
C__inference_dense_7_layer_call_and_return_conditional_losses_157251

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_2_layer_call_fn_157119

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1549932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
(__inference_decoder_layer_call_fn_155791
input_4
unknown:	�
	unknown_0:	�#
	unknown_1:@@
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1557722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4
�
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_157145

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_157110

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_5_layer_call_and_return_conditional_losses_157180

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_155506

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2
Relu�
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�&
�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_157465

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_154993

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�z
�
C__inference_decoder_layer_call_and_return_conditional_losses_157057

inputs9
&dense_7_matmul_readvariableop_resource:	�6
'dense_7_biasadd_readvariableop_resource:	�U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@@@
2conv2d_transpose_3_biasadd_readvariableop_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_4_biasadd_readvariableop_resource: U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_5_biasadd_readvariableop_resource:
identity��)conv2d_transpose_3/BiasAdd/ReadVariableOp�2conv2d_transpose_3/conv2d_transpose/ReadVariableOp�)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_7/MatMul�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_7/Relul
reshape_1/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape�
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack�
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1�
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape_1/Reshape/shape/3�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape�
reshape_1/ReshapeReshapedense_7/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
reshape_1/Reshape~
conv2d_transpose_3/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape�
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack�
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1�
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2�
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/3�
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack�
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack�
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1�
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2�
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1�
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose�
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp�
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_transpose_3/BiasAdd�
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_transpose_3/Relu�
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shape�
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stack�
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1�
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slicez
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/1z
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_4/stack/3�
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stack�
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stack�
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1�
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2%
#conv2d_transpose_4/conv2d_transpose�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOp�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_transpose_4/BiasAdd�
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_transpose_4/Relu�
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape�
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack�
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1�
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slicez
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/1z
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/2z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/3�
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack�
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_1/stack�
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1�
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2d_transpose_5/BiasAdd�
conv2d_transpose_5/SigmoidSigmoid#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
conv2d_transpose_5/Sigmoid�
IdentityIdentityconv2d_transpose_5/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_decoder_layer_call_and_return_conditional_losses_155947
input_4!
dense_7_155925:	�
dense_7_155927:	�3
conv2d_transpose_3_155931:@@'
conv2d_transpose_3_155933:@3
conv2d_transpose_4_155936: @'
conv2d_transpose_4_155938: 3
conv2d_transpose_5_155941: '
conv2d_transpose_5_155943:
identity��*conv2d_transpose_3/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_7_155925dense_7_155927*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1556622!
dense_7/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1556822
reshape_1/PartitionedCall�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_155931conv2d_transpose_3_155933*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1557072,
*conv2d_transpose_3/StatefulPartitionedCall�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_155936conv2d_transpose_4_155938*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1557362,
*conv2d_transpose_4/StatefulPartitionedCall�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_155941conv2d_transpose_5_155943*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1557652,
*conv2d_transpose_5/StatefulPartitionedCall�
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4
�
�
$__inference_vae_layer_call_fn_156672

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	�

unknown_10:	�$

unknown_11:@@

unknown_12:@$

unknown_13: @

unknown_14: $

unknown_15: 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vae_layer_call_and_return_conditional_losses_1560202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_155707

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_6_layer_call_and_return_conditional_losses_157199

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_156357
	vae_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	�

unknown_10:	�$

unknown_11:@@

unknown_12:@$

unknown_13: @

unknown_14: $

unknown_15: 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	vae_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_1549752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:���������
#
_user_specified_name	vae_input
��
�
?__inference_vae_layer_call_and_return_conditional_losses_156631

inputsI
/encoder_conv2d_2_conv2d_readvariableop_resource: >
0encoder_conv2d_2_biasadd_readvariableop_resource: I
/encoder_conv2d_3_conv2d_readvariableop_resource: @>
0encoder_conv2d_3_biasadd_readvariableop_resource:@A
.encoder_dense_4_matmul_readvariableop_resource:	�=
/encoder_dense_4_biasadd_readvariableop_resource:@
.encoder_dense_5_matmul_readvariableop_resource:=
/encoder_dense_5_biasadd_readvariableop_resource:@
.encoder_dense_6_matmul_readvariableop_resource:=
/encoder_dense_6_biasadd_readvariableop_resource:A
.decoder_dense_7_matmul_readvariableop_resource:	�>
/decoder_dense_7_biasadd_readvariableop_resource:	�]
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@@H
:decoder_conv2d_transpose_3_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @H
:decoder_conv2d_transpose_4_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource: H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identity��1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp�:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp�:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp�1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp�:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp�&decoder/dense_7/BiasAdd/ReadVariableOp�%decoder/dense_7/MatMul/ReadVariableOp�'encoder/conv2d_2/BiasAdd/ReadVariableOp�&encoder/conv2d_2/Conv2D/ReadVariableOp�'encoder/conv2d_3/BiasAdd/ReadVariableOp�&encoder/conv2d_3/Conv2D/ReadVariableOp�&encoder/dense_4/BiasAdd/ReadVariableOp�%encoder/dense_4/MatMul/ReadVariableOp�&encoder/dense_5/BiasAdd/ReadVariableOp�%encoder/dense_5/MatMul/ReadVariableOp�&encoder/dense_6/BiasAdd/ReadVariableOp�%encoder/dense_6/MatMul/ReadVariableOp�
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&encoder/conv2d_2/Conv2D/ReadVariableOp�
encoder/conv2d_2/Conv2DConv2Dinputs.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
encoder/conv2d_2/Conv2D�
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'encoder/conv2d_2/BiasAdd/ReadVariableOp�
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
encoder/conv2d_2/BiasAdd�
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
encoder/conv2d_2/Relu�
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02(
&encoder/conv2d_3/Conv2D/ReadVariableOp�
encoder/conv2d_3/Conv2DConv2D#encoder/conv2d_2/Relu:activations:0.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
encoder/conv2d_3/Conv2D�
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_3/BiasAdd/ReadVariableOp�
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
encoder/conv2d_3/BiasAdd�
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
encoder/conv2d_3/Relu�
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
encoder/flatten_1/Const�
encoder/flatten_1/ReshapeReshape#encoder/conv2d_3/Relu:activations:0 encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
encoder/flatten_1/Reshape�
%encoder/dense_4/MatMul/ReadVariableOpReadVariableOp.encoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02'
%encoder/dense_4/MatMul/ReadVariableOp�
encoder/dense_4/MatMulMatMul"encoder/flatten_1/Reshape:output:0-encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_4/MatMul�
&encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_4/BiasAdd/ReadVariableOp�
encoder/dense_4/BiasAddBiasAdd encoder/dense_4/MatMul:product:0.encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_4/BiasAdd�
encoder/dense_4/ReluRelu encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
encoder/dense_4/Relu�
%encoder/dense_5/MatMul/ReadVariableOpReadVariableOp.encoder_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%encoder/dense_5/MatMul/ReadVariableOp�
encoder/dense_5/MatMulMatMul"encoder/dense_4/Relu:activations:0-encoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_5/MatMul�
&encoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_5/BiasAdd/ReadVariableOp�
encoder/dense_5/BiasAddBiasAdd encoder/dense_5/MatMul:product:0.encoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_5/BiasAdd�
%encoder/dense_6/MatMul/ReadVariableOpReadVariableOp.encoder_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%encoder/dense_6/MatMul/ReadVariableOp�
encoder/dense_6/MatMulMatMul"encoder/dense_4/Relu:activations:0-encoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_6/MatMul�
&encoder/dense_6/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_6/BiasAdd/ReadVariableOp�
encoder/dense_6/BiasAddBiasAdd encoder/dense_6/MatMul:product:0.encoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_6/BiasAddy
encoder/sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
encoder/sampling_1/mul/x�
encoder/sampling_1/mulMul!encoder/sampling_1/mul/x:output:0 encoder/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
encoder/sampling_1/mul�
encoder/sampling_1/ExpExpencoder/sampling_1/mul:z:0*
T0*'
_output_shapes
:���������2
encoder/sampling_1/Exp�
encoder/sampling_1/ShapeShape encoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling_1/Shape�
&encoder/sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&encoder/sampling_1/strided_slice/stack�
(encoder/sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling_1/strided_slice/stack_1�
(encoder/sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling_1/strided_slice/stack_2�
 encoder/sampling_1/strided_sliceStridedSlice!encoder/sampling_1/Shape:output:0/encoder/sampling_1/strided_slice/stack:output:01encoder/sampling_1/strided_slice/stack_1:output:01encoder/sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 encoder/sampling_1/strided_slice�
encoder/sampling_1/Shape_1Shape encoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling_1/Shape_1�
(encoder/sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling_1/strided_slice_1/stack�
*encoder/sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*encoder/sampling_1/strided_slice_1/stack_1�
*encoder/sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*encoder/sampling_1/strided_slice_1/stack_2�
"encoder/sampling_1/strided_slice_1StridedSlice#encoder/sampling_1/Shape_1:output:01encoder/sampling_1/strided_slice_1/stack:output:03encoder/sampling_1/strided_slice_1/stack_1:output:03encoder/sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"encoder/sampling_1/strided_slice_1�
&encoder/sampling_1/random_normal/shapePack)encoder/sampling_1/strided_slice:output:0+encoder/sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&encoder/sampling_1/random_normal/shape�
%encoder/sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%encoder/sampling_1/random_normal/mean�
'encoder/sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'encoder/sampling_1/random_normal/stddev�
5encoder/sampling_1/random_normal/RandomStandardNormalRandomStandardNormal/encoder/sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:������������������*
dtype027
5encoder/sampling_1/random_normal/RandomStandardNormal�
$encoder/sampling_1/random_normal/mulMul>encoder/sampling_1/random_normal/RandomStandardNormal:output:00encoder/sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:������������������2&
$encoder/sampling_1/random_normal/mul�
 encoder/sampling_1/random_normalAddV2(encoder/sampling_1/random_normal/mul:z:0.encoder/sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:������������������2"
 encoder/sampling_1/random_normal�
encoder/sampling_1/mul_1Mulencoder/sampling_1/Exp:y:0$encoder/sampling_1/random_normal:z:0*
T0*'
_output_shapes
:���������2
encoder/sampling_1/mul_1�
encoder/sampling_1/addAddV2 encoder/dense_5/BiasAdd:output:0encoder/sampling_1/mul_1:z:0*
T0*'
_output_shapes
:���������2
encoder/sampling_1/add�
%decoder/dense_7/MatMul/ReadVariableOpReadVariableOp.decoder_dense_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02'
%decoder/dense_7/MatMul/ReadVariableOp�
decoder/dense_7/MatMulMatMulencoder/sampling_1/add:z:0-decoder/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
decoder/dense_7/MatMul�
&decoder/dense_7/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&decoder/dense_7/BiasAdd/ReadVariableOp�
decoder/dense_7/BiasAddBiasAdd decoder/dense_7/MatMul:product:0.decoder/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
decoder/dense_7/BiasAdd�
decoder/dense_7/ReluRelu decoder/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
decoder/dense_7/Relu�
decoder/reshape_1/ShapeShape"decoder/dense_7/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_1/Shape�
%decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder/reshape_1/strided_slice/stack�
'decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_1�
'decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_2�
decoder/reshape_1/strided_sliceStridedSlice decoder/reshape_1/Shape:output:0.decoder/reshape_1/strided_slice/stack:output:00decoder/reshape_1/strided_slice/stack_1:output:00decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder/reshape_1/strided_slice�
!decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/1�
!decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/2�
!decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2#
!decoder/reshape_1/Reshape/shape/3�
decoder/reshape_1/Reshape/shapePack(decoder/reshape_1/strided_slice:output:0*decoder/reshape_1/Reshape/shape/1:output:0*decoder/reshape_1/Reshape/shape/2:output:0*decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/reshape_1/Reshape/shape�
decoder/reshape_1/ReshapeReshape"decoder/dense_7/Relu:activations:0(decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
decoder/reshape_1/Reshape�
 decoder/conv2d_transpose_3/ShapeShape"decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/Shape�
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_3/strided_slice/stack�
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_1�
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_2�
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_3/strided_slice�
"decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/stack/1�
"decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/stack/2�
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2$
"decoder/conv2d_transpose_3/stack/3�
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0+decoder/conv2d_transpose_3/stack/1:output:0+decoder/conv2d_transpose_3/stack/2:output:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/stack�
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_3/strided_slice_1/stack�
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_1�
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_2�
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_3/strided_slice_1�
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0"decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_3/conv2d_transpose�
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp�
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2$
"decoder/conv2d_transpose_3/BiasAdd�
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2!
decoder/conv2d_transpose_3/Relu�
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/Shape�
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_4/strided_slice/stack�
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_1�
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_2�
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_4/strided_slice�
"decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/stack/1�
"decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/stack/2�
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"decoder/conv2d_transpose_4/stack/3�
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0+decoder/conv2d_transpose_4/stack/1:output:0+decoder/conv2d_transpose_4/stack/2:output:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/stack�
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_4/strided_slice_1/stack�
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_1�
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_2�
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_4/strided_slice_1�
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp�
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2-
+decoder/conv2d_transpose_4/conv2d_transpose�
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp�
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2$
"decoder/conv2d_transpose_4/BiasAdd�
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2!
decoder/conv2d_transpose_4/Relu�
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/Shape�
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_5/strided_slice/stack�
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_1�
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_2�
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_5/strided_slice�
"decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/1�
"decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/2�
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/3�
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0+decoder/conv2d_transpose_5/stack/1:output:0+decoder/conv2d_transpose_5/stack/2:output:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/stack�
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_5/strided_slice_1/stack�
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_1�
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_2�
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_5/strided_slice_1�
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02<
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp�
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_5/conv2d_transpose�
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp�
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2$
"decoder/conv2d_transpose_5/BiasAdd�
"decoder/conv2d_transpose_5/SigmoidSigmoid+decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������2$
"decoder/conv2d_transpose_5/Sigmoid�
IdentityIdentity&decoder/conv2d_transpose_5/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp2^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp'^decoder/dense_7/BiasAdd/ReadVariableOp&^decoder/dense_7/MatMul/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp'^encoder/dense_4/BiasAdd/ReadVariableOp&^encoder/dense_4/MatMul/ReadVariableOp'^encoder/dense_5/BiasAdd/ReadVariableOp&^encoder/dense_5/MatMul/ReadVariableOp'^encoder/dense_6/BiasAdd/ReadVariableOp&^encoder/dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2P
&decoder/dense_7/BiasAdd/ReadVariableOp&decoder/dense_7/BiasAdd/ReadVariableOp2N
%decoder/dense_7/MatMul/ReadVariableOp%decoder/dense_7/MatMul/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp2P
&encoder/dense_4/BiasAdd/ReadVariableOp&encoder/dense_4/BiasAdd/ReadVariableOp2N
%encoder/dense_4/MatMul/ReadVariableOp%encoder/dense_4/MatMul/ReadVariableOp2P
&encoder/dense_5/BiasAdd/ReadVariableOp&encoder/dense_5/BiasAdd/ReadVariableOp2N
%encoder/dense_5/MatMul/ReadVariableOp%encoder/dense_5/MatMul/ReadVariableOp2P
&encoder/dense_6/BiasAdd/ReadVariableOp&encoder/dense_6/BiasAdd/ReadVariableOp2N
%encoder/dense_6/MatMul/ReadVariableOp%encoder/dense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_157274

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_3_layer_call_fn_157139

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1550102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
(__inference_dense_6_layer_call_fn_157208

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1550672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�P
�
C__inference_encoder_layer_call_and_return_conditional_losses_156775

inputsA
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource: @6
(conv2d_3_biasadd_readvariableop_resource:@9
&dense_4_matmul_readvariableop_resource:	�5
'dense_4_biasadd_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity

identity_1

identity_2��conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_2/Relu�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_3/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten_1/Const�
flatten_1/ReshapeReshapeconv2d_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_4/Relu�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/BiasAdd�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMuldense_4/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/MatMul�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/BiasAddi
sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling_1/mul/x�
sampling_1/mulMulsampling_1/mul/x:output:0dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sampling_1/mulm
sampling_1/ExpExpsampling_1/mul:z:0*
T0*'
_output_shapes
:���������2
sampling_1/Expl
sampling_1/ShapeShapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape�
sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
sampling_1/strided_slice/stack�
 sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_1�
 sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_2�
sampling_1/strided_sliceStridedSlicesampling_1/Shape:output:0'sampling_1/strided_slice/stack:output:0)sampling_1/strided_slice/stack_1:output:0)sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_slicep
sampling_1/Shape_1Shapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape_1�
 sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice_1/stack�
"sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_1�
"sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_2�
sampling_1/strided_slice_1StridedSlicesampling_1/Shape_1:output:0)sampling_1/strided_slice_1/stack:output:0+sampling_1/strided_slice_1/stack_1:output:0+sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_slice_1�
sampling_1/random_normal/shapePack!sampling_1/strided_slice:output:0#sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2 
sampling_1/random_normal/shape�
sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling_1/random_normal/mean�
sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
sampling_1/random_normal/stddev�
-sampling_1/random_normal/RandomStandardNormalRandomStandardNormal'sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:������������������*
dtype02/
-sampling_1/random_normal/RandomStandardNormal�
sampling_1/random_normal/mulMul6sampling_1/random_normal/RandomStandardNormal:output:0(sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:������������������2
sampling_1/random_normal/mul�
sampling_1/random_normalAddV2 sampling_1/random_normal/mul:z:0&sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:������������������2
sampling_1/random_normal�
sampling_1/mul_1Mulsampling_1/Exp:y:0sampling_1/random_normal:z:0*
T0*'
_output_shapes
:���������2
sampling_1/mul_1�
sampling_1/addAddV2dense_5/BiasAdd:output:0sampling_1/mul_1:z:0*
T0*'
_output_shapes
:���������2
sampling_1/addm
IdentityIdentitysampling_1/add:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identityw

Identity_1Identitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_1w

Identity_2Identitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_2�
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_155418

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
$__inference_vae_layer_call_fn_156713

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	�

unknown_10:	�$

unknown_11:@@

unknown_12:@$

unknown_13: @

unknown_14: $

unknown_15: 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vae_layer_call_and_return_conditional_losses_1561462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_154975
	vae_inputM
3vae_encoder_conv2d_2_conv2d_readvariableop_resource: B
4vae_encoder_conv2d_2_biasadd_readvariableop_resource: M
3vae_encoder_conv2d_3_conv2d_readvariableop_resource: @B
4vae_encoder_conv2d_3_biasadd_readvariableop_resource:@E
2vae_encoder_dense_4_matmul_readvariableop_resource:	�A
3vae_encoder_dense_4_biasadd_readvariableop_resource:D
2vae_encoder_dense_5_matmul_readvariableop_resource:A
3vae_encoder_dense_5_biasadd_readvariableop_resource:D
2vae_encoder_dense_6_matmul_readvariableop_resource:A
3vae_encoder_dense_6_biasadd_readvariableop_resource:E
2vae_decoder_dense_7_matmul_readvariableop_resource:	�B
3vae_decoder_dense_7_biasadd_readvariableop_resource:	�a
Gvae_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@@L
>vae_decoder_conv2d_transpose_3_biasadd_readvariableop_resource:@a
Gvae_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @L
>vae_decoder_conv2d_transpose_4_biasadd_readvariableop_resource: a
Gvae_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource: L
>vae_decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identity��5vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp�>vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�5vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp�>vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp�5vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp�>vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp�*vae/decoder/dense_7/BiasAdd/ReadVariableOp�)vae/decoder/dense_7/MatMul/ReadVariableOp�+vae/encoder/conv2d_2/BiasAdd/ReadVariableOp�*vae/encoder/conv2d_2/Conv2D/ReadVariableOp�+vae/encoder/conv2d_3/BiasAdd/ReadVariableOp�*vae/encoder/conv2d_3/Conv2D/ReadVariableOp�*vae/encoder/dense_4/BiasAdd/ReadVariableOp�)vae/encoder/dense_4/MatMul/ReadVariableOp�*vae/encoder/dense_5/BiasAdd/ReadVariableOp�)vae/encoder/dense_5/MatMul/ReadVariableOp�*vae/encoder/dense_6/BiasAdd/ReadVariableOp�)vae/encoder/dense_6/MatMul/ReadVariableOp�
*vae/encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3vae_encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*vae/encoder/conv2d_2/Conv2D/ReadVariableOp�
vae/encoder/conv2d_2/Conv2DConv2D	vae_input2vae/encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
vae/encoder/conv2d_2/Conv2D�
+vae/encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4vae_encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+vae/encoder/conv2d_2/BiasAdd/ReadVariableOp�
vae/encoder/conv2d_2/BiasAddBiasAdd$vae/encoder/conv2d_2/Conv2D:output:03vae/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
vae/encoder/conv2d_2/BiasAdd�
vae/encoder/conv2d_2/ReluRelu%vae/encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
vae/encoder/conv2d_2/Relu�
*vae/encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3vae_encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*vae/encoder/conv2d_3/Conv2D/ReadVariableOp�
vae/encoder/conv2d_3/Conv2DConv2D'vae/encoder/conv2d_2/Relu:activations:02vae/encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
vae/encoder/conv2d_3/Conv2D�
+vae/encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4vae_encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+vae/encoder/conv2d_3/BiasAdd/ReadVariableOp�
vae/encoder/conv2d_3/BiasAddBiasAdd$vae/encoder/conv2d_3/Conv2D:output:03vae/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
vae/encoder/conv2d_3/BiasAdd�
vae/encoder/conv2d_3/ReluRelu%vae/encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
vae/encoder/conv2d_3/Relu�
vae/encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
vae/encoder/flatten_1/Const�
vae/encoder/flatten_1/ReshapeReshape'vae/encoder/conv2d_3/Relu:activations:0$vae/encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
vae/encoder/flatten_1/Reshape�
)vae/encoder/dense_4/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02+
)vae/encoder/dense_4/MatMul/ReadVariableOp�
vae/encoder/dense_4/MatMulMatMul&vae/encoder/flatten_1/Reshape:output:01vae/encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
vae/encoder/dense_4/MatMul�
*vae/encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*vae/encoder/dense_4/BiasAdd/ReadVariableOp�
vae/encoder/dense_4/BiasAddBiasAdd$vae/encoder/dense_4/MatMul:product:02vae/encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
vae/encoder/dense_4/BiasAdd�
vae/encoder/dense_4/ReluRelu$vae/encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
vae/encoder/dense_4/Relu�
)vae/encoder/dense_5/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)vae/encoder/dense_5/MatMul/ReadVariableOp�
vae/encoder/dense_5/MatMulMatMul&vae/encoder/dense_4/Relu:activations:01vae/encoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
vae/encoder/dense_5/MatMul�
*vae/encoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*vae/encoder/dense_5/BiasAdd/ReadVariableOp�
vae/encoder/dense_5/BiasAddBiasAdd$vae/encoder/dense_5/MatMul:product:02vae/encoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
vae/encoder/dense_5/BiasAdd�
)vae/encoder/dense_6/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)vae/encoder/dense_6/MatMul/ReadVariableOp�
vae/encoder/dense_6/MatMulMatMul&vae/encoder/dense_4/Relu:activations:01vae/encoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
vae/encoder/dense_6/MatMul�
*vae/encoder/dense_6/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*vae/encoder/dense_6/BiasAdd/ReadVariableOp�
vae/encoder/dense_6/BiasAddBiasAdd$vae/encoder/dense_6/MatMul:product:02vae/encoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
vae/encoder/dense_6/BiasAdd�
vae/encoder/sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
vae/encoder/sampling_1/mul/x�
vae/encoder/sampling_1/mulMul%vae/encoder/sampling_1/mul/x:output:0$vae/encoder/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
vae/encoder/sampling_1/mul�
vae/encoder/sampling_1/ExpExpvae/encoder/sampling_1/mul:z:0*
T0*'
_output_shapes
:���������2
vae/encoder/sampling_1/Exp�
vae/encoder/sampling_1/ShapeShape$vae/encoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
vae/encoder/sampling_1/Shape�
*vae/encoder/sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*vae/encoder/sampling_1/strided_slice/stack�
,vae/encoder/sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,vae/encoder/sampling_1/strided_slice/stack_1�
,vae/encoder/sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,vae/encoder/sampling_1/strided_slice/stack_2�
$vae/encoder/sampling_1/strided_sliceStridedSlice%vae/encoder/sampling_1/Shape:output:03vae/encoder/sampling_1/strided_slice/stack:output:05vae/encoder/sampling_1/strided_slice/stack_1:output:05vae/encoder/sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$vae/encoder/sampling_1/strided_slice�
vae/encoder/sampling_1/Shape_1Shape$vae/encoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2 
vae/encoder/sampling_1/Shape_1�
,vae/encoder/sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,vae/encoder/sampling_1/strided_slice_1/stack�
.vae/encoder/sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.vae/encoder/sampling_1/strided_slice_1/stack_1�
.vae/encoder/sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.vae/encoder/sampling_1/strided_slice_1/stack_2�
&vae/encoder/sampling_1/strided_slice_1StridedSlice'vae/encoder/sampling_1/Shape_1:output:05vae/encoder/sampling_1/strided_slice_1/stack:output:07vae/encoder/sampling_1/strided_slice_1/stack_1:output:07vae/encoder/sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&vae/encoder/sampling_1/strided_slice_1�
*vae/encoder/sampling_1/random_normal/shapePack-vae/encoder/sampling_1/strided_slice:output:0/vae/encoder/sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2,
*vae/encoder/sampling_1/random_normal/shape�
)vae/encoder/sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)vae/encoder/sampling_1/random_normal/mean�
+vae/encoder/sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+vae/encoder/sampling_1/random_normal/stddev�
9vae/encoder/sampling_1/random_normal/RandomStandardNormalRandomStandardNormal3vae/encoder/sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:������������������*
dtype02;
9vae/encoder/sampling_1/random_normal/RandomStandardNormal�
(vae/encoder/sampling_1/random_normal/mulMulBvae/encoder/sampling_1/random_normal/RandomStandardNormal:output:04vae/encoder/sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:������������������2*
(vae/encoder/sampling_1/random_normal/mul�
$vae/encoder/sampling_1/random_normalAddV2,vae/encoder/sampling_1/random_normal/mul:z:02vae/encoder/sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:������������������2&
$vae/encoder/sampling_1/random_normal�
vae/encoder/sampling_1/mul_1Mulvae/encoder/sampling_1/Exp:y:0(vae/encoder/sampling_1/random_normal:z:0*
T0*'
_output_shapes
:���������2
vae/encoder/sampling_1/mul_1�
vae/encoder/sampling_1/addAddV2$vae/encoder/dense_5/BiasAdd:output:0 vae/encoder/sampling_1/mul_1:z:0*
T0*'
_output_shapes
:���������2
vae/encoder/sampling_1/add�
)vae/decoder/dense_7/MatMul/ReadVariableOpReadVariableOp2vae_decoder_dense_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02+
)vae/decoder/dense_7/MatMul/ReadVariableOp�
vae/decoder/dense_7/MatMulMatMulvae/encoder/sampling_1/add:z:01vae/decoder/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
vae/decoder/dense_7/MatMul�
*vae/decoder/dense_7/BiasAdd/ReadVariableOpReadVariableOp3vae_decoder_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*vae/decoder/dense_7/BiasAdd/ReadVariableOp�
vae/decoder/dense_7/BiasAddBiasAdd$vae/decoder/dense_7/MatMul:product:02vae/decoder/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
vae/decoder/dense_7/BiasAdd�
vae/decoder/dense_7/ReluRelu$vae/decoder/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
vae/decoder/dense_7/Relu�
vae/decoder/reshape_1/ShapeShape&vae/decoder/dense_7/Relu:activations:0*
T0*
_output_shapes
:2
vae/decoder/reshape_1/Shape�
)vae/decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)vae/decoder/reshape_1/strided_slice/stack�
+vae/decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+vae/decoder/reshape_1/strided_slice/stack_1�
+vae/decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+vae/decoder/reshape_1/strided_slice/stack_2�
#vae/decoder/reshape_1/strided_sliceStridedSlice$vae/decoder/reshape_1/Shape:output:02vae/decoder/reshape_1/strided_slice/stack:output:04vae/decoder/reshape_1/strided_slice/stack_1:output:04vae/decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#vae/decoder/reshape_1/strided_slice�
%vae/decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%vae/decoder/reshape_1/Reshape/shape/1�
%vae/decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%vae/decoder/reshape_1/Reshape/shape/2�
%vae/decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%vae/decoder/reshape_1/Reshape/shape/3�
#vae/decoder/reshape_1/Reshape/shapePack,vae/decoder/reshape_1/strided_slice:output:0.vae/decoder/reshape_1/Reshape/shape/1:output:0.vae/decoder/reshape_1/Reshape/shape/2:output:0.vae/decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#vae/decoder/reshape_1/Reshape/shape�
vae/decoder/reshape_1/ReshapeReshape&vae/decoder/dense_7/Relu:activations:0,vae/decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
vae/decoder/reshape_1/Reshape�
$vae/decoder/conv2d_transpose_3/ShapeShape&vae/decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2&
$vae/decoder/conv2d_transpose_3/Shape�
2vae/decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2vae/decoder/conv2d_transpose_3/strided_slice/stack�
4vae/decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4vae/decoder/conv2d_transpose_3/strided_slice/stack_1�
4vae/decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4vae/decoder/conv2d_transpose_3/strided_slice/stack_2�
,vae/decoder/conv2d_transpose_3/strided_sliceStridedSlice-vae/decoder/conv2d_transpose_3/Shape:output:0;vae/decoder/conv2d_transpose_3/strided_slice/stack:output:0=vae/decoder/conv2d_transpose_3/strided_slice/stack_1:output:0=vae/decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,vae/decoder/conv2d_transpose_3/strided_slice�
&vae/decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae/decoder/conv2d_transpose_3/stack/1�
&vae/decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae/decoder/conv2d_transpose_3/stack/2�
&vae/decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2(
&vae/decoder/conv2d_transpose_3/stack/3�
$vae/decoder/conv2d_transpose_3/stackPack5vae/decoder/conv2d_transpose_3/strided_slice:output:0/vae/decoder/conv2d_transpose_3/stack/1:output:0/vae/decoder/conv2d_transpose_3/stack/2:output:0/vae/decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2&
$vae/decoder/conv2d_transpose_3/stack�
4vae/decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4vae/decoder/conv2d_transpose_3/strided_slice_1/stack�
6vae/decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6vae/decoder/conv2d_transpose_3/strided_slice_1/stack_1�
6vae/decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6vae/decoder/conv2d_transpose_3/strided_slice_1/stack_2�
.vae/decoder/conv2d_transpose_3/strided_slice_1StridedSlice-vae/decoder/conv2d_transpose_3/stack:output:0=vae/decoder/conv2d_transpose_3/strided_slice_1/stack:output:0?vae/decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0?vae/decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.vae/decoder/conv2d_transpose_3/strided_slice_1�
>vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpGvae_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02@
>vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�
/vae/decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput-vae/decoder/conv2d_transpose_3/stack:output:0Fvae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0&vae/decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
21
/vae/decoder/conv2d_transpose_3/conv2d_transpose�
5vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp>vae_decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp�
&vae/decoder/conv2d_transpose_3/BiasAddBiasAdd8vae/decoder/conv2d_transpose_3/conv2d_transpose:output:0=vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2(
&vae/decoder/conv2d_transpose_3/BiasAdd�
#vae/decoder/conv2d_transpose_3/ReluRelu/vae/decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2%
#vae/decoder/conv2d_transpose_3/Relu�
$vae/decoder/conv2d_transpose_4/ShapeShape1vae/decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2&
$vae/decoder/conv2d_transpose_4/Shape�
2vae/decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2vae/decoder/conv2d_transpose_4/strided_slice/stack�
4vae/decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4vae/decoder/conv2d_transpose_4/strided_slice/stack_1�
4vae/decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4vae/decoder/conv2d_transpose_4/strided_slice/stack_2�
,vae/decoder/conv2d_transpose_4/strided_sliceStridedSlice-vae/decoder/conv2d_transpose_4/Shape:output:0;vae/decoder/conv2d_transpose_4/strided_slice/stack:output:0=vae/decoder/conv2d_transpose_4/strided_slice/stack_1:output:0=vae/decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,vae/decoder/conv2d_transpose_4/strided_slice�
&vae/decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae/decoder/conv2d_transpose_4/stack/1�
&vae/decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae/decoder/conv2d_transpose_4/stack/2�
&vae/decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2(
&vae/decoder/conv2d_transpose_4/stack/3�
$vae/decoder/conv2d_transpose_4/stackPack5vae/decoder/conv2d_transpose_4/strided_slice:output:0/vae/decoder/conv2d_transpose_4/stack/1:output:0/vae/decoder/conv2d_transpose_4/stack/2:output:0/vae/decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2&
$vae/decoder/conv2d_transpose_4/stack�
4vae/decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4vae/decoder/conv2d_transpose_4/strided_slice_1/stack�
6vae/decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6vae/decoder/conv2d_transpose_4/strided_slice_1/stack_1�
6vae/decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6vae/decoder/conv2d_transpose_4/strided_slice_1/stack_2�
.vae/decoder/conv2d_transpose_4/strided_slice_1StridedSlice-vae/decoder/conv2d_transpose_4/stack:output:0=vae/decoder/conv2d_transpose_4/strided_slice_1/stack:output:0?vae/decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0?vae/decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.vae/decoder/conv2d_transpose_4/strided_slice_1�
>vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpGvae_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02@
>vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp�
/vae/decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput-vae/decoder/conv2d_transpose_4/stack:output:0Fvae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:01vae/decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
21
/vae/decoder/conv2d_transpose_4/conv2d_transpose�
5vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp>vae_decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp�
&vae/decoder/conv2d_transpose_4/BiasAddBiasAdd8vae/decoder/conv2d_transpose_4/conv2d_transpose:output:0=vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2(
&vae/decoder/conv2d_transpose_4/BiasAdd�
#vae/decoder/conv2d_transpose_4/ReluRelu/vae/decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2%
#vae/decoder/conv2d_transpose_4/Relu�
$vae/decoder/conv2d_transpose_5/ShapeShape1vae/decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2&
$vae/decoder/conv2d_transpose_5/Shape�
2vae/decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2vae/decoder/conv2d_transpose_5/strided_slice/stack�
4vae/decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4vae/decoder/conv2d_transpose_5/strided_slice/stack_1�
4vae/decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4vae/decoder/conv2d_transpose_5/strided_slice/stack_2�
,vae/decoder/conv2d_transpose_5/strided_sliceStridedSlice-vae/decoder/conv2d_transpose_5/Shape:output:0;vae/decoder/conv2d_transpose_5/strided_slice/stack:output:0=vae/decoder/conv2d_transpose_5/strided_slice/stack_1:output:0=vae/decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,vae/decoder/conv2d_transpose_5/strided_slice�
&vae/decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae/decoder/conv2d_transpose_5/stack/1�
&vae/decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae/decoder/conv2d_transpose_5/stack/2�
&vae/decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae/decoder/conv2d_transpose_5/stack/3�
$vae/decoder/conv2d_transpose_5/stackPack5vae/decoder/conv2d_transpose_5/strided_slice:output:0/vae/decoder/conv2d_transpose_5/stack/1:output:0/vae/decoder/conv2d_transpose_5/stack/2:output:0/vae/decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2&
$vae/decoder/conv2d_transpose_5/stack�
4vae/decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4vae/decoder/conv2d_transpose_5/strided_slice_1/stack�
6vae/decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6vae/decoder/conv2d_transpose_5/strided_slice_1/stack_1�
6vae/decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6vae/decoder/conv2d_transpose_5/strided_slice_1/stack_2�
.vae/decoder/conv2d_transpose_5/strided_slice_1StridedSlice-vae/decoder/conv2d_transpose_5/stack:output:0=vae/decoder/conv2d_transpose_5/strided_slice_1/stack:output:0?vae/decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0?vae/decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.vae/decoder/conv2d_transpose_5/strided_slice_1�
>vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpGvae_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02@
>vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp�
/vae/decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput-vae/decoder/conv2d_transpose_5/stack:output:0Fvae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:01vae/decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
21
/vae/decoder/conv2d_transpose_5/conv2d_transpose�
5vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp>vae_decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp�
&vae/decoder/conv2d_transpose_5/BiasAddBiasAdd8vae/decoder/conv2d_transpose_5/conv2d_transpose:output:0=vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2(
&vae/decoder/conv2d_transpose_5/BiasAdd�
&vae/decoder/conv2d_transpose_5/SigmoidSigmoid/vae/decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������2(
&vae/decoder/conv2d_transpose_5/Sigmoid�
IdentityIdentity*vae/decoder/conv2d_transpose_5/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp6^vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?^vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp6^vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?^vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp6^vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?^vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp+^vae/decoder/dense_7/BiasAdd/ReadVariableOp*^vae/decoder/dense_7/MatMul/ReadVariableOp,^vae/encoder/conv2d_2/BiasAdd/ReadVariableOp+^vae/encoder/conv2d_2/Conv2D/ReadVariableOp,^vae/encoder/conv2d_3/BiasAdd/ReadVariableOp+^vae/encoder/conv2d_3/Conv2D/ReadVariableOp+^vae/encoder/dense_4/BiasAdd/ReadVariableOp*^vae/encoder/dense_4/MatMul/ReadVariableOp+^vae/encoder/dense_5/BiasAdd/ReadVariableOp*^vae/encoder/dense_5/MatMul/ReadVariableOp+^vae/encoder/dense_6/BiasAdd/ReadVariableOp*^vae/encoder/dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2n
5vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp5vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2�
>vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp>vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2n
5vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp5vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2�
>vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp>vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2n
5vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp5vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2�
>vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp>vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2X
*vae/decoder/dense_7/BiasAdd/ReadVariableOp*vae/decoder/dense_7/BiasAdd/ReadVariableOp2V
)vae/decoder/dense_7/MatMul/ReadVariableOp)vae/decoder/dense_7/MatMul/ReadVariableOp2Z
+vae/encoder/conv2d_2/BiasAdd/ReadVariableOp+vae/encoder/conv2d_2/BiasAdd/ReadVariableOp2X
*vae/encoder/conv2d_2/Conv2D/ReadVariableOp*vae/encoder/conv2d_2/Conv2D/ReadVariableOp2Z
+vae/encoder/conv2d_3/BiasAdd/ReadVariableOp+vae/encoder/conv2d_3/BiasAdd/ReadVariableOp2X
*vae/encoder/conv2d_3/Conv2D/ReadVariableOp*vae/encoder/conv2d_3/Conv2D/ReadVariableOp2X
*vae/encoder/dense_4/BiasAdd/ReadVariableOp*vae/encoder/dense_4/BiasAdd/ReadVariableOp2V
)vae/encoder/dense_4/MatMul/ReadVariableOp)vae/encoder/dense_4/MatMul/ReadVariableOp2X
*vae/encoder/dense_5/BiasAdd/ReadVariableOp*vae/encoder/dense_5/BiasAdd/ReadVariableOp2V
)vae/encoder/dense_5/MatMul/ReadVariableOp)vae/encoder/dense_5/MatMul/ReadVariableOp2X
*vae/encoder/dense_6/BiasAdd/ReadVariableOp*vae/encoder/dense_6/BiasAdd/ReadVariableOp2V
)vae/encoder/dense_6/MatMul/ReadVariableOp)vae/encoder/dense_6/MatMul/ReadVariableOp:Z V
/
_output_shapes
:���������
#
_user_specified_name	vae_input
�
�
?__inference_vae_layer_call_and_return_conditional_losses_156020

inputs(
encoder_155979: 
encoder_155981: (
encoder_155983: @
encoder_155985:@!
encoder_155987:	�
encoder_155989: 
encoder_155991:
encoder_155993: 
encoder_155995:
encoder_155997:!
decoder_156002:	�
decoder_156004:	�(
decoder_156006:@@
decoder_156008:@(
decoder_156010: @
decoder_156012: (
decoder_156014: 
decoder_156016:
identity��decoder/StatefulPartitionedCall�encoder/StatefulPartitionedCall�
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_155979encoder_155981encoder_155983encoder_155985encoder_155987encoder_155989encoder_155991encoder_155993encoder_155995encoder_155997*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1551042!
encoder/StatefulPartitionedCall�
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_156002decoder_156004decoder_156006decoder_156008decoder_156010decoder_156012decoder_156014decoder_156016*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1557722!
decoder/StatefulPartitionedCall�
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
?__inference_vae_layer_call_and_return_conditional_losses_156270
	vae_input(
encoder_156229: 
encoder_156231: (
encoder_156233: @
encoder_156235:@!
encoder_156237:	�
encoder_156239: 
encoder_156241:
encoder_156243: 
encoder_156245:
encoder_156247:!
decoder_156252:	�
decoder_156254:	�(
decoder_156256:@@
decoder_156258:@(
decoder_156260: @
decoder_156262: (
decoder_156264: 
decoder_156266:
identity��decoder/StatefulPartitionedCall�encoder/StatefulPartitionedCall�
encoder/StatefulPartitionedCallStatefulPartitionedCall	vae_inputencoder_156229encoder_156231encoder_156233encoder_156235encoder_156237encoder_156239encoder_156241encoder_156243encoder_156245encoder_156247*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1551042!
encoder/StatefulPartitionedCall�
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_156252decoder_156254decoder_156256decoder_156258decoder_156260decoder_156262decoder_156264decoder_156266*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1557722!
decoder/StatefulPartitionedCall�
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Z V
/
_output_shapes
:���������
#
_user_specified_name	vae_input
�

�
C__inference_dense_6_layer_call_and_return_conditional_losses_155067

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_156895

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1552582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_155736

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3�
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� 2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
F
*__inference_reshape_1_layer_call_fn_157279

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1556822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_155022

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_156866

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1551042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_155131
input_3!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1551042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
�
s
F__inference_sampling_1_layer_call_and_return_conditional_losses_155099

inputs
inputs_1
identity�S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:���������2
ExpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:������������������*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:������������������2
random_normal/mul�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:������������������2
random_normalc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:���������2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_7_layer_call_fn_157260

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1556622
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
u
F__inference_sampling_1_layer_call_and_return_conditional_losses_157234
inputs_0
inputs_1
identity�S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:���������2
ExpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:������������������*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:������������������2
random_normal/mul�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:������������������2
random_normalc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_157413

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3�
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� 2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_3_layer_call_fn_157355

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1557072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_5_layer_call_and_return_conditional_losses_155051

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�(
�
C__inference_encoder_layer_call_and_return_conditional_losses_155258

inputs)
conv2d_2_155228: 
conv2d_2_155230: )
conv2d_3_155233: @
conv2d_3_155235:@!
dense_4_155239:	�
dense_4_155241: 
dense_5_155244:
dense_5_155246: 
dense_6_155249:
dense_6_155251:
identity

identity_1

identity_2�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�"sampling_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_155228conv2d_2_155230*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1549932"
 conv2d_2/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_155233conv2d_3_155235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1550102"
 conv2d_3/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1550222
flatten_1/PartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_155239dense_4_155241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1550352!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_155244dense_5_155246*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1550512!
dense_5/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_6_155249dense_6_155251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1550672!
dense_6/StatefulPartitionedCall�
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_1550992$
"sampling_1/StatefulPartitionedCall�
IdentityIdentity+sampling_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_2�
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
(__inference_decoder_layer_call_fn_155922
input_4
unknown:	�
	unknown_0:	�#
	unknown_1:@@
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1558822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4
�
�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_155765

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������2	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_157130

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
(__inference_dense_4_layer_call_fn_157170

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1550352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_5_layer_call_fn_157498

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1555942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

�
(__inference_decoder_layer_call_fn_157078

inputs
unknown:	�
	unknown_0:	�#
	unknown_1:@@
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1557722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_157389

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2
Relu�
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�(
�
C__inference_encoder_layer_call_and_return_conditional_losses_155104

inputs)
conv2d_2_154994: 
conv2d_2_154996: )
conv2d_3_155011: @
conv2d_3_155013:@!
dense_4_155036:	�
dense_4_155038: 
dense_5_155052:
dense_5_155054: 
dense_6_155068:
dense_6_155070:
identity

identity_1

identity_2�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�"sampling_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_154994conv2d_2_154996*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1549932"
 conv2d_2/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_155011conv2d_3_155013*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1550102"
 conv2d_3/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1550222
flatten_1/PartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_155036dense_4_155038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1550352!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_155052dense_5_155054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1550512!
dense_5/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_6_155068dense_6_155070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1550672!
dense_6/StatefulPartitionedCall�
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_1550992$
"sampling_1/StatefulPartitionedCall�
IdentityIdentity+sampling_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_2�
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_157161

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_3_layer_call_fn_157346

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1554182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
t
+__inference_sampling_1_layer_call_fn_157240
inputs_0
inputs_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_1550992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
?__inference_vae_layer_call_and_return_conditional_losses_156146

inputs(
encoder_156105: 
encoder_156107: (
encoder_156109: @
encoder_156111:@!
encoder_156113:	�
encoder_156115: 
encoder_156117:
encoder_156119: 
encoder_156121:
encoder_156123:!
decoder_156128:	�
decoder_156130:	�(
decoder_156132:@@
decoder_156134:@(
decoder_156136: @
decoder_156138: (
decoder_156140: 
decoder_156142:
identity��decoder/StatefulPartitionedCall�encoder/StatefulPartitionedCall�
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_156105encoder_156107encoder_156109encoder_156111encoder_156113encoder_156115encoder_156117encoder_156119encoder_156121encoder_156123*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1552582!
encoder/StatefulPartitionedCall�
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_156128decoder_156130decoder_156132decoder_156134decoder_156136decoder_156138decoder_156140decoder_156142*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1558822!
decoder/StatefulPartitionedCall�
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_4_layer_call_fn_157431

inputs!
unknown: @
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1557362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_4_layer_call_fn_157422

inputs!
unknown: @
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1555062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_155314
input_3!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1552582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
�P
�
C__inference_encoder_layer_call_and_return_conditional_losses_156837

inputsA
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource: @6
(conv2d_3_biasadd_readvariableop_resource:@9
&dense_4_matmul_readvariableop_resource:	�5
'dense_4_biasadd_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:
identity

identity_1

identity_2��conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_2/Relu�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_3/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten_1/Const�
flatten_1/ReshapeReshapeconv2d_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_4/Relu�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/BiasAdd�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMuldense_4/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/MatMul�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/BiasAddi
sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling_1/mul/x�
sampling_1/mulMulsampling_1/mul/x:output:0dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sampling_1/mulm
sampling_1/ExpExpsampling_1/mul:z:0*
T0*'
_output_shapes
:���������2
sampling_1/Expl
sampling_1/ShapeShapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape�
sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
sampling_1/strided_slice/stack�
 sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_1�
 sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_2�
sampling_1/strided_sliceStridedSlicesampling_1/Shape:output:0'sampling_1/strided_slice/stack:output:0)sampling_1/strided_slice/stack_1:output:0)sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_slicep
sampling_1/Shape_1Shapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape_1�
 sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice_1/stack�
"sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_1�
"sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_2�
sampling_1/strided_slice_1StridedSlicesampling_1/Shape_1:output:0)sampling_1/strided_slice_1/stack:output:0+sampling_1/strided_slice_1/stack_1:output:0+sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_slice_1�
sampling_1/random_normal/shapePack!sampling_1/strided_slice:output:0#sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2 
sampling_1/random_normal/shape�
sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling_1/random_normal/mean�
sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
sampling_1/random_normal/stddev�
-sampling_1/random_normal/RandomStandardNormalRandomStandardNormal'sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:������������������*
dtype02/
-sampling_1/random_normal/RandomStandardNormal�
sampling_1/random_normal/mulMul6sampling_1/random_normal/RandomStandardNormal:output:0(sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:������������������2
sampling_1/random_normal/mul�
sampling_1/random_normalAddV2 sampling_1/random_normal/mul:z:0&sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:������������������2
sampling_1/random_normal�
sampling_1/mul_1Mulsampling_1/Exp:y:0sampling_1/random_normal:z:0*
T0*'
_output_shapes
:���������2
sampling_1/mul_1�
sampling_1/addAddV2dense_5/BiasAdd:output:0sampling_1/mul_1:z:0*
T0*'
_output_shapes
:���������2
sampling_1/addm
IdentityIdentitysampling_1/add:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identityw

Identity_1Identitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_1w

Identity_2Identitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_2�
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
F
*__inference_flatten_1_layer_call_fn_157150

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1550222
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_155035

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�(
�
C__inference_encoder_layer_call_and_return_conditional_losses_155347
input_3)
conv2d_2_155317: 
conv2d_2_155319: )
conv2d_3_155322: @
conv2d_3_155324:@!
dense_4_155328:	�
dense_4_155330: 
dense_5_155333:
dense_5_155335: 
dense_6_155338:
dense_6_155340:
identity

identity_1

identity_2�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�"sampling_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_2_155317conv2d_2_155319*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1549932"
 conv2d_2/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_155322conv2d_3_155324*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1550102"
 conv2d_3/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1550222
flatten_1/PartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_155328dense_4_155330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1550352!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_155333dense_5_155335*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1550512!
dense_5/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_6_155338dense_6_155340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1550672!
dense_6/StatefulPartitionedCall�
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_1550992$
"sampling_1/StatefulPartitionedCall�
IdentityIdentity+sampling_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_2�
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
�
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_155682

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_5_layer_call_fn_157189

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1550512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_decoder_layer_call_and_return_conditional_losses_155972
input_4!
dense_7_155950:	�
dense_7_155952:	�3
conv2d_transpose_3_155956:@@'
conv2d_transpose_3_155958:@3
conv2d_transpose_4_155961: @'
conv2d_transpose_4_155963: 3
conv2d_transpose_5_155966: '
conv2d_transpose_5_155968:
identity��*conv2d_transpose_3/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_7_155950dense_7_155952*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1556622!
dense_7/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1556822
reshape_1/PartitionedCall�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_155956conv2d_transpose_3_155958*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1557072,
*conv2d_transpose_3/StatefulPartitionedCall�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_155961conv2d_transpose_4_155963*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1557362,
*conv2d_transpose_4/StatefulPartitionedCall�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_155966conv2d_transpose_5_155968*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1557652,
*conv2d_transpose_5/StatefulPartitionedCall�
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4
�

�
(__inference_decoder_layer_call_fn_157099

inputs
unknown:	�
	unknown_0:	�#
	unknown_1:@@
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1558822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
?__inference_vae_layer_call_and_return_conditional_losses_156494

inputsI
/encoder_conv2d_2_conv2d_readvariableop_resource: >
0encoder_conv2d_2_biasadd_readvariableop_resource: I
/encoder_conv2d_3_conv2d_readvariableop_resource: @>
0encoder_conv2d_3_biasadd_readvariableop_resource:@A
.encoder_dense_4_matmul_readvariableop_resource:	�=
/encoder_dense_4_biasadd_readvariableop_resource:@
.encoder_dense_5_matmul_readvariableop_resource:=
/encoder_dense_5_biasadd_readvariableop_resource:@
.encoder_dense_6_matmul_readvariableop_resource:=
/encoder_dense_6_biasadd_readvariableop_resource:A
.decoder_dense_7_matmul_readvariableop_resource:	�>
/decoder_dense_7_biasadd_readvariableop_resource:	�]
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@@H
:decoder_conv2d_transpose_3_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @H
:decoder_conv2d_transpose_4_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource: H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identity��1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp�:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp�:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp�1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp�:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp�&decoder/dense_7/BiasAdd/ReadVariableOp�%decoder/dense_7/MatMul/ReadVariableOp�'encoder/conv2d_2/BiasAdd/ReadVariableOp�&encoder/conv2d_2/Conv2D/ReadVariableOp�'encoder/conv2d_3/BiasAdd/ReadVariableOp�&encoder/conv2d_3/Conv2D/ReadVariableOp�&encoder/dense_4/BiasAdd/ReadVariableOp�%encoder/dense_4/MatMul/ReadVariableOp�&encoder/dense_5/BiasAdd/ReadVariableOp�%encoder/dense_5/MatMul/ReadVariableOp�&encoder/dense_6/BiasAdd/ReadVariableOp�%encoder/dense_6/MatMul/ReadVariableOp�
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&encoder/conv2d_2/Conv2D/ReadVariableOp�
encoder/conv2d_2/Conv2DConv2Dinputs.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
encoder/conv2d_2/Conv2D�
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'encoder/conv2d_2/BiasAdd/ReadVariableOp�
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
encoder/conv2d_2/BiasAdd�
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
encoder/conv2d_2/Relu�
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02(
&encoder/conv2d_3/Conv2D/ReadVariableOp�
encoder/conv2d_3/Conv2DConv2D#encoder/conv2d_2/Relu:activations:0.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
encoder/conv2d_3/Conv2D�
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_3/BiasAdd/ReadVariableOp�
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
encoder/conv2d_3/BiasAdd�
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
encoder/conv2d_3/Relu�
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
encoder/flatten_1/Const�
encoder/flatten_1/ReshapeReshape#encoder/conv2d_3/Relu:activations:0 encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
encoder/flatten_1/Reshape�
%encoder/dense_4/MatMul/ReadVariableOpReadVariableOp.encoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02'
%encoder/dense_4/MatMul/ReadVariableOp�
encoder/dense_4/MatMulMatMul"encoder/flatten_1/Reshape:output:0-encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_4/MatMul�
&encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_4/BiasAdd/ReadVariableOp�
encoder/dense_4/BiasAddBiasAdd encoder/dense_4/MatMul:product:0.encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_4/BiasAdd�
encoder/dense_4/ReluRelu encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
encoder/dense_4/Relu�
%encoder/dense_5/MatMul/ReadVariableOpReadVariableOp.encoder_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%encoder/dense_5/MatMul/ReadVariableOp�
encoder/dense_5/MatMulMatMul"encoder/dense_4/Relu:activations:0-encoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_5/MatMul�
&encoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_5/BiasAdd/ReadVariableOp�
encoder/dense_5/BiasAddBiasAdd encoder/dense_5/MatMul:product:0.encoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_5/BiasAdd�
%encoder/dense_6/MatMul/ReadVariableOpReadVariableOp.encoder_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%encoder/dense_6/MatMul/ReadVariableOp�
encoder/dense_6/MatMulMatMul"encoder/dense_4/Relu:activations:0-encoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_6/MatMul�
&encoder/dense_6/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_6/BiasAdd/ReadVariableOp�
encoder/dense_6/BiasAddBiasAdd encoder/dense_6/MatMul:product:0.encoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/dense_6/BiasAddy
encoder/sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
encoder/sampling_1/mul/x�
encoder/sampling_1/mulMul!encoder/sampling_1/mul/x:output:0 encoder/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
encoder/sampling_1/mul�
encoder/sampling_1/ExpExpencoder/sampling_1/mul:z:0*
T0*'
_output_shapes
:���������2
encoder/sampling_1/Exp�
encoder/sampling_1/ShapeShape encoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling_1/Shape�
&encoder/sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&encoder/sampling_1/strided_slice/stack�
(encoder/sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling_1/strided_slice/stack_1�
(encoder/sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling_1/strided_slice/stack_2�
 encoder/sampling_1/strided_sliceStridedSlice!encoder/sampling_1/Shape:output:0/encoder/sampling_1/strided_slice/stack:output:01encoder/sampling_1/strided_slice/stack_1:output:01encoder/sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 encoder/sampling_1/strided_slice�
encoder/sampling_1/Shape_1Shape encoder/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling_1/Shape_1�
(encoder/sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling_1/strided_slice_1/stack�
*encoder/sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*encoder/sampling_1/strided_slice_1/stack_1�
*encoder/sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*encoder/sampling_1/strided_slice_1/stack_2�
"encoder/sampling_1/strided_slice_1StridedSlice#encoder/sampling_1/Shape_1:output:01encoder/sampling_1/strided_slice_1/stack:output:03encoder/sampling_1/strided_slice_1/stack_1:output:03encoder/sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"encoder/sampling_1/strided_slice_1�
&encoder/sampling_1/random_normal/shapePack)encoder/sampling_1/strided_slice:output:0+encoder/sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&encoder/sampling_1/random_normal/shape�
%encoder/sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%encoder/sampling_1/random_normal/mean�
'encoder/sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'encoder/sampling_1/random_normal/stddev�
5encoder/sampling_1/random_normal/RandomStandardNormalRandomStandardNormal/encoder/sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:������������������*
dtype027
5encoder/sampling_1/random_normal/RandomStandardNormal�
$encoder/sampling_1/random_normal/mulMul>encoder/sampling_1/random_normal/RandomStandardNormal:output:00encoder/sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:������������������2&
$encoder/sampling_1/random_normal/mul�
 encoder/sampling_1/random_normalAddV2(encoder/sampling_1/random_normal/mul:z:0.encoder/sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:������������������2"
 encoder/sampling_1/random_normal�
encoder/sampling_1/mul_1Mulencoder/sampling_1/Exp:y:0$encoder/sampling_1/random_normal:z:0*
T0*'
_output_shapes
:���������2
encoder/sampling_1/mul_1�
encoder/sampling_1/addAddV2 encoder/dense_5/BiasAdd:output:0encoder/sampling_1/mul_1:z:0*
T0*'
_output_shapes
:���������2
encoder/sampling_1/add�
%decoder/dense_7/MatMul/ReadVariableOpReadVariableOp.decoder_dense_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02'
%decoder/dense_7/MatMul/ReadVariableOp�
decoder/dense_7/MatMulMatMulencoder/sampling_1/add:z:0-decoder/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
decoder/dense_7/MatMul�
&decoder/dense_7/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&decoder/dense_7/BiasAdd/ReadVariableOp�
decoder/dense_7/BiasAddBiasAdd decoder/dense_7/MatMul:product:0.decoder/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
decoder/dense_7/BiasAdd�
decoder/dense_7/ReluRelu decoder/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
decoder/dense_7/Relu�
decoder/reshape_1/ShapeShape"decoder/dense_7/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_1/Shape�
%decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder/reshape_1/strided_slice/stack�
'decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_1�
'decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_1/strided_slice/stack_2�
decoder/reshape_1/strided_sliceStridedSlice decoder/reshape_1/Shape:output:0.decoder/reshape_1/strided_slice/stack:output:00decoder/reshape_1/strided_slice/stack_1:output:00decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder/reshape_1/strided_slice�
!decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/1�
!decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_1/Reshape/shape/2�
!decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2#
!decoder/reshape_1/Reshape/shape/3�
decoder/reshape_1/Reshape/shapePack(decoder/reshape_1/strided_slice:output:0*decoder/reshape_1/Reshape/shape/1:output:0*decoder/reshape_1/Reshape/shape/2:output:0*decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/reshape_1/Reshape/shape�
decoder/reshape_1/ReshapeReshape"decoder/dense_7/Relu:activations:0(decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
decoder/reshape_1/Reshape�
 decoder/conv2d_transpose_3/ShapeShape"decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/Shape�
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_3/strided_slice/stack�
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_1�
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_3/strided_slice/stack_2�
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_3/strided_slice�
"decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/stack/1�
"decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_3/stack/2�
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2$
"decoder/conv2d_transpose_3/stack/3�
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0+decoder/conv2d_transpose_3/stack/1:output:0+decoder/conv2d_transpose_3/stack/2:output:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_3/stack�
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_3/strided_slice_1/stack�
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_1�
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_3/strided_slice_1/stack_2�
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_3/strided_slice_1�
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0"decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_3/conv2d_transpose�
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp�
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2$
"decoder/conv2d_transpose_3/BiasAdd�
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2!
decoder/conv2d_transpose_3/Relu�
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/Shape�
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_4/strided_slice/stack�
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_1�
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_4/strided_slice/stack_2�
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_4/strided_slice�
"decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/stack/1�
"decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_4/stack/2�
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"decoder/conv2d_transpose_4/stack/3�
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0+decoder/conv2d_transpose_4/stack/1:output:0+decoder/conv2d_transpose_4/stack/2:output:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_4/stack�
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_4/strided_slice_1/stack�
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_1�
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_4/strided_slice_1/stack_2�
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_4/strided_slice_1�
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp�
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2-
+decoder/conv2d_transpose_4/conv2d_transpose�
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp�
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2$
"decoder/conv2d_transpose_4/BiasAdd�
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2!
decoder/conv2d_transpose_4/Relu�
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/Shape�
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_5/strided_slice/stack�
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_1�
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_5/strided_slice/stack_2�
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_5/strided_slice�
"decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/1�
"decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/2�
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_5/stack/3�
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0+decoder/conv2d_transpose_5/stack/1:output:0+decoder/conv2d_transpose_5/stack/2:output:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_5/stack�
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_5/strided_slice_1/stack�
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_1�
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_5/strided_slice_1/stack_2�
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_5/strided_slice_1�
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02<
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp�
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_5/conv2d_transpose�
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp�
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2$
"decoder/conv2d_transpose_5/BiasAdd�
"decoder/conv2d_transpose_5/SigmoidSigmoid+decoder/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������2$
"decoder/conv2d_transpose_5/Sigmoid�
IdentityIdentity&decoder/conv2d_transpose_5/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp2^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp'^decoder/dense_7/BiasAdd/ReadVariableOp&^decoder/dense_7/MatMul/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp'^encoder/dense_4/BiasAdd/ReadVariableOp&^encoder/dense_4/MatMul/ReadVariableOp'^encoder/dense_5/BiasAdd/ReadVariableOp&^encoder/dense_5/MatMul/ReadVariableOp'^encoder/dense_6/BiasAdd/ReadVariableOp&^encoder/dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2P
&decoder/dense_7/BiasAdd/ReadVariableOp&decoder/dense_7/BiasAdd/ReadVariableOp2N
%decoder/dense_7/MatMul/ReadVariableOp%decoder/dense_7/MatMul/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp2P
&encoder/dense_4/BiasAdd/ReadVariableOp&encoder/dense_4/BiasAdd/ReadVariableOp2N
%encoder/dense_4/MatMul/ReadVariableOp%encoder/dense_4/MatMul/ReadVariableOp2P
&encoder/dense_5/BiasAdd/ReadVariableOp&encoder/dense_5/BiasAdd/ReadVariableOp2N
%encoder/dense_5/MatMul/ReadVariableOp%encoder/dense_5/MatMul/ReadVariableOp2P
&encoder/dense_6/BiasAdd/ReadVariableOp&encoder/dense_6/BiasAdd/ReadVariableOp2N
%encoder/dense_6/MatMul/ReadVariableOp%encoder/dense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�O
�
"__inference__traced_restore_157648
file_prefix:
 assignvariableop_conv2d_2_kernel: .
 assignvariableop_1_conv2d_2_bias: <
"assignvariableop_2_conv2d_3_kernel: @.
 assignvariableop_3_conv2d_3_bias:@4
!assignvariableop_4_dense_4_kernel:	�-
assignvariableop_5_dense_4_bias:3
!assignvariableop_6_dense_5_kernel:-
assignvariableop_7_dense_5_bias:3
!assignvariableop_8_dense_6_kernel:-
assignvariableop_9_dense_6_bias:5
"assignvariableop_10_dense_7_kernel:	�/
 assignvariableop_11_dense_7_bias:	�G
-assignvariableop_12_conv2d_transpose_3_kernel:@@9
+assignvariableop_13_conv2d_transpose_3_bias:@G
-assignvariableop_14_conv2d_transpose_4_kernel: @9
+assignvariableop_15_conv2d_transpose_4_bias: G
-assignvariableop_16_conv2d_transpose_5_kernel: 9
+assignvariableop_17_conv2d_transpose_5_bias:
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_7_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_conv2d_transpose_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp+assignvariableop_13_conv2d_transpose_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_conv2d_transpose_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp-assignvariableop_16_conv2d_transpose_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_conv2d_transpose_5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18f
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_19�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_157489

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������2	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�&
�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_155594

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�/
�
__inference__traced_save_157584
file_prefix.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : @:@:	�::::::	�:�:@@:@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	�: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::%!

_output_shapes
:	�:!

_output_shapes	
:�:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
�
�
3__inference_conv2d_transpose_5_layer_call_fn_157507

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1557652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�(
�
C__inference_encoder_layer_call_and_return_conditional_losses_155380
input_3)
conv2d_2_155350: 
conv2d_2_155352: )
conv2d_3_155355: @
conv2d_3_155357:@!
dense_4_155361:	�
dense_4_155363: 
dense_5_155366:
dense_5_155368: 
dense_6_155371:
dense_6_155373:
identity

identity_1

identity_2�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�"sampling_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_2_155350conv2d_2_155352*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1549932"
 conv2d_2/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_155355conv2d_3_155357*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1550102"
 conv2d_3/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1550222
flatten_1/PartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_155361dense_4_155363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1550352!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_155366dense_5_155368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1550512!
dense_5/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_6_155371dense_6_155373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1550672!
dense_6/StatefulPartitionedCall�
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_1550992$
"sampling_1/StatefulPartitionedCall�
IdentityIdentity+sampling_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity_2�
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
�
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_157337

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_decoder_layer_call_and_return_conditional_losses_155882

inputs!
dense_7_155860:	�
dense_7_155862:	�3
conv2d_transpose_3_155866:@@'
conv2d_transpose_3_155868:@3
conv2d_transpose_4_155871: @'
conv2d_transpose_4_155873: 3
conv2d_transpose_5_155876: '
conv2d_transpose_5_155878:
identity��*conv2d_transpose_3/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_155860dense_7_155862*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1556622!
dense_7/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1556822
reshape_1/PartitionedCall�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_155866conv2d_transpose_3_155868*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1557072,
*conv2d_transpose_3/StatefulPartitionedCall�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_155871conv2d_transpose_4_155873*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1557362,
*conv2d_transpose_4/StatefulPartitionedCall�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_155876conv2d_transpose_5_155878*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1557652,
*conv2d_transpose_5/StatefulPartitionedCall�
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_7_layer_call_and_return_conditional_losses_155662

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_157313

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
$__inference_vae_layer_call_fn_156059
	vae_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	�

unknown_10:	�$

unknown_11:@@

unknown_12:@$

unknown_13: @

unknown_14: $

unknown_15: 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	vae_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vae_layer_call_and_return_conditional_losses_1560202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:���������
#
_user_specified_name	vae_input
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_155010

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
C__inference_decoder_layer_call_and_return_conditional_losses_155772

inputs!
dense_7_155663:	�
dense_7_155665:	�3
conv2d_transpose_3_155708:@@'
conv2d_transpose_3_155710:@3
conv2d_transpose_4_155737: @'
conv2d_transpose_4_155739: 3
conv2d_transpose_5_155766: '
conv2d_transpose_5_155768:
identity��*conv2d_transpose_3/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_155663dense_7_155665*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_1556622!
dense_7/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1556822
reshape_1/PartitionedCall�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_155708conv2d_transpose_3_155710*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1557072,
*conv2d_transpose_3/StatefulPartitionedCall�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_155737conv2d_transpose_4_155739*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1557362,
*conv2d_transpose_4/StatefulPartitionedCall�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_155766conv2d_transpose_5_155768*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1557652,
*conv2d_transpose_5/StatefulPartitionedCall�
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
?__inference_vae_layer_call_and_return_conditional_losses_156314
	vae_input(
encoder_156273: 
encoder_156275: (
encoder_156277: @
encoder_156279:@!
encoder_156281:	�
encoder_156283: 
encoder_156285:
encoder_156287: 
encoder_156289:
encoder_156291:!
decoder_156296:	�
decoder_156298:	�(
decoder_156300:@@
decoder_156302:@(
decoder_156304: @
decoder_156306: (
decoder_156308: 
decoder_156310:
identity��decoder/StatefulPartitionedCall�encoder/StatefulPartitionedCall�
encoder/StatefulPartitionedCallStatefulPartitionedCall	vae_inputencoder_156273encoder_156275encoder_156277encoder_156279encoder_156281encoder_156283encoder_156285encoder_156287encoder_156289encoder_156291*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1552582!
encoder/StatefulPartitionedCall�
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_156296decoder_156298decoder_156300decoder_156302decoder_156304decoder_156306decoder_156308decoder_156310*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_1558822!
decoder/StatefulPartitionedCall�
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Z V
/
_output_shapes
:���������
#
_user_specified_name	vae_input
�z
�
C__inference_decoder_layer_call_and_return_conditional_losses_156976

inputs9
&dense_7_matmul_readvariableop_resource:	�6
'dense_7_biasadd_readvariableop_resource:	�U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@@@
2conv2d_transpose_3_biasadd_readvariableop_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_4_biasadd_readvariableop_resource: U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_5_biasadd_readvariableop_resource:
identity��)conv2d_transpose_3/BiasAdd/ReadVariableOp�2conv2d_transpose_3/conv2d_transpose/ReadVariableOp�)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_7/MatMul�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_7/Relul
reshape_1/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape�
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack�
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1�
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape_1/Reshape/shape/3�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape�
reshape_1/ReshapeReshapedense_7/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
reshape_1/Reshape~
conv2d_transpose_3/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape�
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack�
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1�
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2�
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/3�
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack�
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack�
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1�
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2�
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1�
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose�
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp�
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_transpose_3/BiasAdd�
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_transpose_3/Relu�
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shape�
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stack�
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1�
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slicez
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/1z
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_4/stack/3�
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stack�
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stack�
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1�
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2%
#conv2d_transpose_4/conv2d_transpose�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOp�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_transpose_4/BiasAdd�
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_transpose_4/Relu�
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape�
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack�
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1�
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slicez
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/1z
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/2z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/3�
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack�
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_1/stack�
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1�
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2d_transpose_5/BiasAdd�
conv2d_transpose_5/SigmoidSigmoid#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
conv2d_transpose_5/Sigmoid�
IdentityIdentityconv2d_transpose_5/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������2

Identity�
NoOpNoOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
	vae_input:
serving_default_vae_input:0���������C
decoder8
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�
	layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
regularization_losses
trainable_variables
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_network
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
regularization_losses
trainable_variables
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_network
 "
trackable_list_wrapper
�
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017"
trackable_list_wrapper
�
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017"
trackable_list_wrapper
�
1layer_regularization_losses
regularization_losses
trainable_variables
2metrics
3layer_metrics
	variables

4layers
5non_trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
"
_tf_keras_input_layer
�

kernel
 bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

!kernel
"bias
:regularization_losses
;trainable_variables
<	variables
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

#kernel
$bias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

%kernel
&bias
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

'kernel
(bias
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
 "
trackable_list_wrapper
f
0
 1
!2
"3
#4
$5
%6
&7
'8
(9"
trackable_list_wrapper
f
0
 1
!2
"3
#4
$5
%6
&7
'8
(9"
trackable_list_wrapper
�
Rlayer_regularization_losses
regularization_losses
trainable_variables
Smetrics
Tlayer_metrics
	variables

Ulayers
Vnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_tf_keras_input_layer
�

)kernel
*bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
[regularization_losses
\trainable_variables
]	variables
^	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

+kernel
,bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

-kernel
.bias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

/kernel
0bias
gregularization_losses
htrainable_variables
i	variables
j	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
 "
trackable_list_wrapper
X
)0
*1
+2
,3
-4
.5
/6
07"
trackable_list_wrapper
X
)0
*1
+2
,3
-4
.5
/6
07"
trackable_list_wrapper
�
klayer_regularization_losses
regularization_losses
trainable_variables
lmetrics
mlayer_metrics
	variables

nlayers
onon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_2/kernel
: 2conv2d_2/bias
):' @2conv2d_3/kernel
:@2conv2d_3/bias
!:	�2dense_4/kernel
:2dense_4/bias
 :2dense_5/kernel
:2dense_5/bias
 :2dense_6/kernel
:2dense_6/bias
!:	�2dense_7/kernel
:�2dense_7/bias
3:1@@2conv2d_transpose_3/kernel
%:#@2conv2d_transpose_3/bias
3:1 @2conv2d_transpose_4/kernel
%:# 2conv2d_transpose_4/bias
3:1 2conv2d_transpose_5/kernel
%:#2conv2d_transpose_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
�
player_regularization_losses
6regularization_losses
qlayer_metrics
rmetrics
7trainable_variables
8	variables

slayers
tnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
�
ulayer_regularization_losses
:regularization_losses
vlayer_metrics
wmetrics
;trainable_variables
<	variables

xlayers
ynon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
zlayer_regularization_losses
>regularization_losses
{layer_metrics
|metrics
?trainable_variables
@	variables

}layers
~non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
�
layer_regularization_losses
Bregularization_losses
�layer_metrics
�metrics
Ctrainable_variables
D	variables
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
�
 �layer_regularization_losses
Fregularization_losses
�layer_metrics
�metrics
Gtrainable_variables
H	variables
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
�
 �layer_regularization_losses
Jregularization_losses
�layer_metrics
�metrics
Ktrainable_variables
L	variables
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
Nregularization_losses
�layer_metrics
�metrics
Otrainable_variables
P	variables
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
	0

1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�
 �layer_regularization_losses
Wregularization_losses
�layer_metrics
�metrics
Xtrainable_variables
Y	variables
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
[regularization_losses
�layer_metrics
�metrics
\trainable_variables
]	variables
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
 �layer_regularization_losses
_regularization_losses
�layer_metrics
�metrics
`trainable_variables
a	variables
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
 �layer_regularization_losses
cregularization_losses
�layer_metrics
�metrics
dtrainable_variables
e	variables
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
�
 �layer_regularization_losses
gregularization_losses
�layer_metrics
�metrics
htrainable_variables
i	variables
�layers
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
?__inference_vae_layer_call_and_return_conditional_losses_156494
?__inference_vae_layer_call_and_return_conditional_losses_156631
?__inference_vae_layer_call_and_return_conditional_losses_156270
?__inference_vae_layer_call_and_return_conditional_losses_156314�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_vae_layer_call_fn_156059
$__inference_vae_layer_call_fn_156672
$__inference_vae_layer_call_fn_156713
$__inference_vae_layer_call_fn_156226�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
!__inference__wrapped_model_154975	vae_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_encoder_layer_call_and_return_conditional_losses_156775
C__inference_encoder_layer_call_and_return_conditional_losses_156837
C__inference_encoder_layer_call_and_return_conditional_losses_155347
C__inference_encoder_layer_call_and_return_conditional_losses_155380�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_encoder_layer_call_fn_155131
(__inference_encoder_layer_call_fn_156866
(__inference_encoder_layer_call_fn_156895
(__inference_encoder_layer_call_fn_155314�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_decoder_layer_call_and_return_conditional_losses_156976
C__inference_decoder_layer_call_and_return_conditional_losses_157057
C__inference_decoder_layer_call_and_return_conditional_losses_155947
C__inference_decoder_layer_call_and_return_conditional_losses_155972�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_decoder_layer_call_fn_155791
(__inference_decoder_layer_call_fn_157078
(__inference_decoder_layer_call_fn_157099
(__inference_decoder_layer_call_fn_155922�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
$__inference_signature_wrapper_156357	vae_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_157110�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_2_layer_call_fn_157119�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_157130�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_3_layer_call_fn_157139�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_flatten_1_layer_call_and_return_conditional_losses_157145�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_flatten_1_layer_call_fn_157150�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_4_layer_call_and_return_conditional_losses_157161�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_4_layer_call_fn_157170�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_5_layer_call_and_return_conditional_losses_157180�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_5_layer_call_fn_157189�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_6_layer_call_and_return_conditional_losses_157199�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_6_layer_call_fn_157208�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_sampling_1_layer_call_and_return_conditional_losses_157234�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_sampling_1_layer_call_fn_157240�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_7_layer_call_and_return_conditional_losses_157251�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_7_layer_call_fn_157260�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_reshape_1_layer_call_and_return_conditional_losses_157274�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_reshape_1_layer_call_fn_157279�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_157313
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_157337�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_conv2d_transpose_3_layer_call_fn_157346
3__inference_conv2d_transpose_3_layer_call_fn_157355�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_157389
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_157413�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_conv2d_transpose_4_layer_call_fn_157422
3__inference_conv2d_transpose_4_layer_call_fn_157431�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_157465
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_157489�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_conv2d_transpose_5_layer_call_fn_157498
3__inference_conv2d_transpose_5_layer_call_fn_157507�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_154975� !"#$%&'()*+,-./0:�7
0�-
+�(
	vae_input���������
� "9�6
4
decoder)�&
decoder����������
D__inference_conv2d_2_layer_call_and_return_conditional_losses_157110l 7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
)__inference_conv2d_2_layer_call_fn_157119_ 7�4
-�*
(�%
inputs���������
� " ���������� �
D__inference_conv2d_3_layer_call_and_return_conditional_losses_157130l!"7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
)__inference_conv2d_3_layer_call_fn_157139_!"7�4
-�*
(�%
inputs��������� 
� " ����������@�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_157313�+,I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_157337l+,7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
3__inference_conv2d_transpose_3_layer_call_fn_157346�+,I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
3__inference_conv2d_transpose_3_layer_call_fn_157355_+,7�4
-�*
(�%
inputs���������@
� " ����������@�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_157389�-.I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+��������������������������� 
� �
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_157413l-.7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0��������� 
� �
3__inference_conv2d_transpose_4_layer_call_fn_157422�-.I�F
?�<
:�7
inputs+���������������������������@
� "2�/+��������������������������� �
3__inference_conv2d_transpose_4_layer_call_fn_157431_-.7�4
-�*
(�%
inputs���������@
� " ���������� �
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_157465�/0I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������
� �
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_157489l/07�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������
� �
3__inference_conv2d_transpose_5_layer_call_fn_157498�/0I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+����������������������������
3__inference_conv2d_transpose_5_layer_call_fn_157507_/07�4
-�*
(�%
inputs��������� 
� " �����������
C__inference_decoder_layer_call_and_return_conditional_losses_155947s)*+,-./08�5
.�+
!�
input_4���������
p 

 
� "-�*
#� 
0���������
� �
C__inference_decoder_layer_call_and_return_conditional_losses_155972s)*+,-./08�5
.�+
!�
input_4���������
p

 
� "-�*
#� 
0���������
� �
C__inference_decoder_layer_call_and_return_conditional_losses_156976r)*+,-./07�4
-�*
 �
inputs���������
p 

 
� "-�*
#� 
0���������
� �
C__inference_decoder_layer_call_and_return_conditional_losses_157057r)*+,-./07�4
-�*
 �
inputs���������
p

 
� "-�*
#� 
0���������
� �
(__inference_decoder_layer_call_fn_155791f)*+,-./08�5
.�+
!�
input_4���������
p 

 
� " �����������
(__inference_decoder_layer_call_fn_155922f)*+,-./08�5
.�+
!�
input_4���������
p

 
� " �����������
(__inference_decoder_layer_call_fn_157078e)*+,-./07�4
-�*
 �
inputs���������
p 

 
� " �����������
(__inference_decoder_layer_call_fn_157099e)*+,-./07�4
-�*
 �
inputs���������
p

 
� " �����������
C__inference_dense_4_layer_call_and_return_conditional_losses_157161]#$0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
(__inference_dense_4_layer_call_fn_157170P#$0�-
&�#
!�
inputs����������
� "�����������
C__inference_dense_5_layer_call_and_return_conditional_losses_157180\%&/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
(__inference_dense_5_layer_call_fn_157189O%&/�,
%�"
 �
inputs���������
� "�����������
C__inference_dense_6_layer_call_and_return_conditional_losses_157199\'(/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
(__inference_dense_6_layer_call_fn_157208O'(/�,
%�"
 �
inputs���������
� "�����������
C__inference_dense_7_layer_call_and_return_conditional_losses_157251])*/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� |
(__inference_dense_7_layer_call_fn_157260P)*/�,
%�"
 �
inputs���������
� "������������
C__inference_encoder_layer_call_and_return_conditional_losses_155347�
 !"#$%&'(@�=
6�3
)�&
input_3���������
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
C__inference_encoder_layer_call_and_return_conditional_losses_155380�
 !"#$%&'(@�=
6�3
)�&
input_3���������
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
C__inference_encoder_layer_call_and_return_conditional_losses_156775�
 !"#$%&'(?�<
5�2
(�%
inputs���������
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
C__inference_encoder_layer_call_and_return_conditional_losses_156837�
 !"#$%&'(?�<
5�2
(�%
inputs���������
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
(__inference_encoder_layer_call_fn_155131�
 !"#$%&'(@�=
6�3
)�&
input_3���������
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
(__inference_encoder_layer_call_fn_155314�
 !"#$%&'(@�=
6�3
)�&
input_3���������
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
(__inference_encoder_layer_call_fn_156866�
 !"#$%&'(?�<
5�2
(�%
inputs���������
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
(__inference_encoder_layer_call_fn_156895�
 !"#$%&'(?�<
5�2
(�%
inputs���������
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
E__inference_flatten_1_layer_call_and_return_conditional_losses_157145a7�4
-�*
(�%
inputs���������@
� "&�#
�
0����������
� �
*__inference_flatten_1_layer_call_fn_157150T7�4
-�*
(�%
inputs���������@
� "������������
E__inference_reshape_1_layer_call_and_return_conditional_losses_157274a0�-
&�#
!�
inputs����������
� "-�*
#� 
0���������@
� �
*__inference_reshape_1_layer_call_fn_157279T0�-
&�#
!�
inputs����������
� " ����������@�
F__inference_sampling_1_layer_call_and_return_conditional_losses_157234�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� �
+__inference_sampling_1_layer_call_fn_157240vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "�����������
$__inference_signature_wrapper_156357� !"#$%&'()*+,-./0G�D
� 
=�:
8
	vae_input+�(
	vae_input���������"9�6
4
decoder)�&
decoder����������
?__inference_vae_layer_call_and_return_conditional_losses_156270� !"#$%&'()*+,-./0B�?
8�5
+�(
	vae_input���������
p 

 
� "-�*
#� 
0���������
� �
?__inference_vae_layer_call_and_return_conditional_losses_156314� !"#$%&'()*+,-./0B�?
8�5
+�(
	vae_input���������
p

 
� "-�*
#� 
0���������
� �
?__inference_vae_layer_call_and_return_conditional_losses_156494� !"#$%&'()*+,-./0?�<
5�2
(�%
inputs���������
p 

 
� "-�*
#� 
0���������
� �
?__inference_vae_layer_call_and_return_conditional_losses_156631� !"#$%&'()*+,-./0?�<
5�2
(�%
inputs���������
p

 
� "-�*
#� 
0���������
� �
$__inference_vae_layer_call_fn_156059z !"#$%&'()*+,-./0B�?
8�5
+�(
	vae_input���������
p 

 
� " �����������
$__inference_vae_layer_call_fn_156226z !"#$%&'()*+,-./0B�?
8�5
+�(
	vae_input���������
p

 
� " �����������
$__inference_vae_layer_call_fn_156672w !"#$%&'()*+,-./0?�<
5�2
(�%
inputs���������
p 

 
� " �����������
$__inference_vae_layer_call_fn_156713w !"#$%&'()*+,-./0?�<
5�2
(�%
inputs���������
p

 
� " ����������