čý
ă
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
delete_old_dirsbool(
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-rc1-69-gc256c071bb28¨
z
layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namelayer_2/kernel
s
"layer_2/kernel/Read/ReadVariableOpReadVariableOplayer_2/kernel* 
_output_shapes
:
*
dtype0
q
layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer_2/bias
j
 layer_2/bias/Read/ReadVariableOpReadVariableOplayer_2/bias*
_output_shapes	
:*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	
*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:
*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*É
valueżBź Bľ
×
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
		keras_api


signatures
 
w
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api


kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
w
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api


kernel
bias
#_self_saveable_object_factories
	variables
 trainable_variables
!regularization_losses
"	keras_api

0
1
2
3

0
1
2
3
 
­
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
 
­
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUElayer_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
­
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
­
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
­
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
 trainable_variables
!regularization_losses
 
#
0
1
2
3
4
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

serving_default_flatten_7_inputPlaceholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙
Ü
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_7_inputlayer_2/kernellayer_2/biasdense_13/kerneldense_13/bias*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *~
_output_shapesl
j:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_1086
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ž
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"layer_2/kernel/Read/ReadVariableOp layer_2/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpConst*
Tin

2*
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
GPU2*0J 8 *&
f!R
__inference__traced_save_1334
Ů
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_2/kernellayer_2/biasdense_13/kerneldense_13/bias*
Tin	
2*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_restore_1356á
 

$__inference_model_layer_call_fn_1128

inputs
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identity

identity_1

identity_2

identity_3

identity_4˘StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *~
_output_shapesl
j:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_965w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť

$__inference_model_layer_call_fn_1005
flatten_7_input
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identity

identity_1

identity_2

identity_3

identity_4˘StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallflatten_7_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *~
_output_shapesl
j:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_965w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_7_input
Ĺ

&__inference_layer_2_layer_call_fn_1221

inputs
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_layer_2_layer_call_and_return_conditional_losses_808p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Č	
ó
A__inference_dense_13_layer_call_and_return_conditional_losses_831

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ź 
ş
?__inference_model_layer_call_and_return_conditional_losses_1057
flatten_7_input 
layer_2_1035:

layer_2_1037:	 
dense_13_1041:	

dense_13_1043:

identity

identity_1

identity_2

identity_3

identity_4˘ dense_13/StatefulPartitionedCall˘"dropout_13/StatefulPartitionedCall˘(kernel/Regularizer/Square/ReadVariableOp˘layer_2/StatefulPartitionedCallÄ
flatten_7/PartitionedCallPartitionedCallflatten_7_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_7_layer_call_and_return_conditional_losses_789
layer_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0layer_2_1035layer_2_1037*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_layer_2_layer_call_and_return_conditional_losses_808ď
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_13_layer_call_and_return_conditional_losses_897
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_13_1041dense_13_1043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_13_layer_call_and_return_conditional_losses_831w
(kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer_2_1035* 
_output_shapes
:
*
dtype0
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
i
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityflatten_7_input^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙t

Identity_1Identity"flatten_7/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z

Identity_2Identity(layer_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙}

Identity_3Identity+dropout_13/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z

Identity_4Identity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
NoOpNoOp!^dense_13/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall)^kernel/Regularizer/Square/ReadVariableOp ^layer_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall:` \
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_7_input


"__inference_signature_wrapper_1086
flatten_7_input
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identity

identity_1

identity_2

identity_3

identity_4˘StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallflatten_7_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *~
_output_shapesl
j:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__wrapped_model_776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_7_input
É	
ô
B__inference_dense_13_layer_call_and_return_conditional_losses_1284

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
 

$__inference_model_layer_call_fn_1107

inputs
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identity

identity_1

identity_2

identity_3

identity_4˘StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *~
_output_shapesl
j:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_848w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs



__inference_loss_fn_0_1295E
1kernel_regularizer_square_readvariableop_resource:

identity˘(kernel/Regularizer/Square/ReadVariableOp
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype0
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
i
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
"
ľ
?__inference_model_layer_call_and_return_conditional_losses_1158

inputs:
&layer_2_matmul_readvariableop_resource:
6
'layer_2_biasadd_readvariableop_resource:	:
'dense_13_matmul_readvariableop_resource:	
6
(dense_13_biasadd_readvariableop_resource:

identity

identity_1

identity_2

identity_3

identity_4˘dense_13/BiasAdd/ReadVariableOp˘dense_13/MatMul/ReadVariableOp˘(kernel/Regularizer/Square/ReadVariableOp˘layer_2/BiasAdd/ReadVariableOp˘layer_2/MatMul/ReadVariableOp`
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  q
flatten_7/ReshapeReshapeinputsflatten_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
layer_2/MatMul/ReadVariableOpReadVariableOp&layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
layer_2/MatMulMatMulflatten_7/Reshape:output:0%layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
layer_2/BiasAddBiasAddlayer_2/MatMul:product:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
layer_2/ReluRelulayer_2/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
dropout_13/IdentityIdentitylayer_2/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense_13/MatMulMatMuldropout_13/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

(kernel/Regularizer/Square/ReadVariableOpReadVariableOp&layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
i
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentityinputs^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙l

Identity_1Identityflatten_7/Reshape:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l

Identity_2Identitylayer_2/Relu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n

Identity_3Identitydropout_13/Identity:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j

Identity_4Identitydense_13/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp^layer_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2>
layer_2/MatMul/ReadVariableOplayer_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ą

@__inference_layer_2_layer_call_and_return_conditional_losses_808

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp˘(kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
i
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ű
b
D__inference_dropout_13_layer_call_and_return_conditional_losses_1253

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ü)
ľ
?__inference_model_layer_call_and_return_conditional_losses_1195

inputs:
&layer_2_matmul_readvariableop_resource:
6
'layer_2_biasadd_readvariableop_resource:	:
'dense_13_matmul_readvariableop_resource:	
6
(dense_13_biasadd_readvariableop_resource:

identity

identity_1

identity_2

identity_3

identity_4˘dense_13/BiasAdd/ReadVariableOp˘dense_13/MatMul/ReadVariableOp˘(kernel/Regularizer/Square/ReadVariableOp˘layer_2/BiasAdd/ReadVariableOp˘layer_2/MatMul/ReadVariableOp`
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  q
flatten_7/ReshapeReshapeinputsflatten_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
layer_2/MatMul/ReadVariableOpReadVariableOp&layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
layer_2/MatMulMatMulflatten_7/Reshape:output:0%layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
layer_2/BiasAddBiasAddlayer_2/MatMul:product:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
layer_2/ReluRelulayer_2/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_13/dropout/MulMullayer_2/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
dropout_13/dropout/ShapeShapelayer_2/Relu:activations:0*
T0*
_output_shapes
:Ł
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0f
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Č
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense_13/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

(kernel/Regularizer/Square/ReadVariableOpReadVariableOp&layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
i
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentityinputs^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙l

Identity_1Identityflatten_7/Reshape:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l

Identity_2Identitylayer_2/Relu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n

Identity_3Identitydropout_13/dropout/Mul_1:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j

Identity_4Identitydense_13/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp^layer_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2>
layer_2/MatMul/ReadVariableOplayer_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ţ
Ź
>__inference_model_layer_call_and_return_conditional_losses_965

inputs
layer_2_943:

layer_2_945:	
dense_13_949:	

dense_13_951:

identity

identity_1

identity_2

identity_3

identity_4˘ dense_13/StatefulPartitionedCall˘"dropout_13/StatefulPartitionedCall˘(kernel/Regularizer/Square/ReadVariableOp˘layer_2/StatefulPartitionedCallť
flatten_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_7_layer_call_and_return_conditional_losses_789
layer_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0layer_2_943layer_2_945*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_layer_2_layer_call_and_return_conditional_losses_808ď
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_13_layer_call_and_return_conditional_losses_897
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_13_949dense_13_951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_13_layer_call_and_return_conditional_losses_831v
(kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer_2_943* 
_output_shapes
:
*
dtype0
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
i
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentityinputs^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙t

Identity_1Identity"flatten_7/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z

Identity_2Identity(layer_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙}

Identity_3Identity+dropout_13/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z

Identity_4Identity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
NoOpNoOp!^dense_13/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall)^kernel/Regularizer/Square/ReadVariableOp ^layer_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ł
E
)__inference_dropout_13_layer_call_fn_1243

inputs
identity˛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_13_layer_call_and_return_conditional_losses_819a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ę

>__inference_model_layer_call_and_return_conditional_losses_848

inputs
layer_2_809:

layer_2_811:	
dense_13_832:	

dense_13_834:

identity

identity_1

identity_2

identity_3

identity_4˘ dense_13/StatefulPartitionedCall˘(kernel/Regularizer/Square/ReadVariableOp˘layer_2/StatefulPartitionedCallť
flatten_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_7_layer_call_and_return_conditional_losses_789
layer_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0layer_2_809layer_2_811*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_layer_2_layer_call_and_return_conditional_losses_808ß
dropout_13/PartitionedCallPartitionedCall(layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_13_layer_call_and_return_conditional_losses_819
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_13_832dense_13_834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_13_layer_call_and_return_conditional_losses_831v
(kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer_2_809* 
_output_shapes
:
*
dtype0
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
i
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentityinputs^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙t

Identity_1Identity"flatten_7/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z

Identity_2Identity(layer_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u

Identity_3Identity#dropout_13/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z

Identity_4Identity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
NoOpNoOp!^dense_13/StatefulPartitionedCall)^kernel/Regularizer/Square/ReadVariableOp ^layer_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ř

?__inference_model_layer_call_and_return_conditional_losses_1031
flatten_7_input 
layer_2_1009:

layer_2_1011:	 
dense_13_1015:	

dense_13_1017:

identity

identity_1

identity_2

identity_3

identity_4˘ dense_13/StatefulPartitionedCall˘(kernel/Regularizer/Square/ReadVariableOp˘layer_2/StatefulPartitionedCallÄ
flatten_7/PartitionedCallPartitionedCallflatten_7_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_7_layer_call_and_return_conditional_losses_789
layer_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0layer_2_1009layer_2_1011*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_layer_2_layer_call_and_return_conditional_losses_808ß
dropout_13/PartitionedCallPartitionedCall(layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_13_layer_call_and_return_conditional_losses_819
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_13_1015dense_13_1017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_13_layer_call_and_return_conditional_losses_831w
(kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer_2_1009* 
_output_shapes
:
*
dtype0
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
i
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityflatten_7_input^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙t

Identity_1Identity"flatten_7/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z

Identity_2Identity(layer_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u

Identity_3Identity#dropout_13/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z

Identity_4Identity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
NoOpNoOp!^dense_13/StatefulPartitionedCall)^kernel/Regularizer/Square/ReadVariableOp ^layer_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall:` \
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_7_input
ő
b
)__inference_dropout_13_layer_call_fn_1248

inputs
identity˘StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_13_layer_call_and_return_conditional_losses_897p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ż
D
(__inference_flatten_7_layer_call_fn_1200

inputs
identityą
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_7_layer_call_and_return_conditional_losses_789a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˛
 
A__inference_layer_2_layer_call_and_return_conditional_losses_1238

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp˘(kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
i
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ú
a
C__inference_dropout_13_layer_call_and_return_conditional_losses_819

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ă

'__inference_dense_13_layer_call_fn_1274

inputs
unknown:	

	unknown_0:

identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_13_layer_call_and_return_conditional_losses_831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Č
ä
 __inference__traced_restore_1356
file_prefix3
assignvariableop_layer_2_kernel:
.
assignvariableop_1_layer_2_bias:	5
"assignvariableop_2_dense_13_kernel:	
.
 assignvariableop_3_dense_13_bias:


identity_5˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3é
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHz
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B ˇ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_layer_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_13_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_13_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ź

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_5IdentityIdentity_4:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

¤
__inference__traced_save_1334
file_prefix-
)savev2_layer_2_kernel_read_readvariableop+
'savev2_layer_2_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ć
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHw
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B Ţ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_layer_2_kernel_read_readvariableop'savev2_layer_2_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*;
_input_shapes*
(: :
::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: 
ş

#__inference_model_layer_call_fn_867
flatten_7_input
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identity

identity_1

identity_2

identity_3

identity_4˘StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallflatten_7_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *~
_output_shapesl
j:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_848w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_7_input
¨
˘
__inference__wrapped_model_776
flatten_7_input@
,model_layer_2_matmul_readvariableop_resource:
<
-model_layer_2_biasadd_readvariableop_resource:	@
-model_dense_13_matmul_readvariableop_resource:	
<
.model_dense_13_biasadd_readvariableop_resource:

identity

identity_1

identity_2

identity_3

identity_4˘%model/dense_13/BiasAdd/ReadVariableOp˘$model/dense_13/MatMul/ReadVariableOp˘$model/layer_2/BiasAdd/ReadVariableOp˘#model/layer_2/MatMul/ReadVariableOpf
model/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  
model/flatten_7/ReshapeReshapeflatten_7_inputmodel/flatten_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#model/layer_2/MatMul/ReadVariableOpReadVariableOp,model_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
model/layer_2/MatMulMatMul model/flatten_7/Reshape:output:0+model/layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
$model/layer_2/BiasAdd/ReadVariableOpReadVariableOp-model_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ą
model/layer_2/BiasAddBiasAddmodel/layer_2/MatMul:product:0,model/layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
model/layer_2/ReluRelumodel/layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
model/dropout_13/IdentityIdentity model/layer_2/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
$model/dense_13/MatMul/ReadVariableOpReadVariableOp-model_dense_13_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0Ł
model/dense_13/MatMulMatMul"model/dropout_13/Identity:output:0,model/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

%model/dense_13/BiasAdd/ReadVariableOpReadVariableOp.model_dense_13_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ł
model/dense_13/BiasAddBiasAddmodel/dense_13/MatMul:product:0-model/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
IdentityIdentitymodel/dense_13/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t

Identity_1Identity"model/dropout_13/Identity:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity model/flatten_7/Reshape:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h

Identity_3Identityflatten_7_input^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_4Identity model/layer_2/Relu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙â
NoOpNoOp&^model/dense_13/BiasAdd/ReadVariableOp%^model/dense_13/MatMul/ReadVariableOp%^model/layer_2/BiasAdd/ReadVariableOp$^model/layer_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙: : : : 2N
%model/dense_13/BiasAdd/ReadVariableOp%model/dense_13/BiasAdd/ReadVariableOp2L
$model/dense_13/MatMul/ReadVariableOp$model/dense_13/MatMul/ReadVariableOp2L
$model/layer_2/BiasAdd/ReadVariableOp$model/layer_2/BiasAdd/ReadVariableOp2J
#model/layer_2/MatMul/ReadVariableOp#model/layer_2/MatMul/ReadVariableOp:` \
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_7_input
ú	
c
D__inference_dropout_13_layer_call_and_return_conditional_losses_1265

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĺ
_
C__inference_flatten_7_layer_call_and_return_conditional_losses_1206

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ů	
b
C__inference_dropout_13_layer_call_and_return_conditional_losses_897

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ä
^
B__inference_flatten_7_layer_call_and_return_conditional_losses_789

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ď
serving_defaultť
S
flatten_7_input@
!serving_default_flatten_7_input:0˙˙˙˙˙˙˙˙˙<
dense_130
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙
?

dropout_131
StatefulPartitionedCall:1˙˙˙˙˙˙˙˙˙>
	flatten_71
StatefulPartitionedCall:2˙˙˙˙˙˙˙˙˙K
flatten_7_input8
StatefulPartitionedCall:3˙˙˙˙˙˙˙˙˙<
layer_21
StatefulPartitionedCall:4˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:Łd
É
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
		keras_api


signatures
<__call__
*=&call_and_return_all_conditional_losses
>_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
Ę
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
ŕ

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
Ę
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
ŕ

kernel
bias
#_self_saveable_object_factories
	variables
 trainable_variables
!regularization_losses
"	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
G0"
trackable_list_wrapper
Ę
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
<__call__
>_default_save_signature
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
,
Hserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
": 
2layer_2/kernel
:2layer_2/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
G0"
trackable_list_wrapper
­
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
": 	
2dense_13/kernel
:
2dense_13/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
 trainable_variables
!regularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
C
0
1
2
3
4"
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
'
G0"
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
Ý2Ú
#__inference_model_layer_call_fn_867
$__inference_model_layer_call_fn_1107
$__inference_model_layer_call_fn_1128
$__inference_model_layer_call_fn_1005Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ę2Ç
?__inference_model_layer_call_and_return_conditional_losses_1158
?__inference_model_layer_call_and_return_conditional_losses_1195
?__inference_model_layer_call_and_return_conditional_losses_1031
?__inference_model_layer_call_and_return_conditional_losses_1057Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŃBÎ
__inference__wrapped_model_776flatten_7_input"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_flatten_7_layer_call_fn_1200˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_flatten_7_layer_call_and_return_conditional_losses_1206˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_layer_2_layer_call_fn_1221˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_layer_2_layer_call_and_return_conditional_losses_1238˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
)__inference_dropout_13_layer_call_fn_1243
)__inference_dropout_13_layer_call_fn_1248´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ć2Ă
D__inference_dropout_13_layer_call_and_return_conditional_losses_1253
D__inference_dropout_13_layer_call_and_return_conditional_losses_1265´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ń2Î
'__inference_dense_13_layer_call_fn_1274˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ě2é
B__inference_dense_13_layer_call_and_return_conditional_losses_1284˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ą2Ž
__inference_loss_fn_0_1295
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ŃBÎ
"__inference_signature_wrapper_1086flatten_7_input"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ˙
__inference__wrapped_model_776Ü@˘=
6˘3
1.
flatten_7_input˙˙˙˙˙˙˙˙˙
Ş "Ş
.
dense_13"
dense_13˙˙˙˙˙˙˙˙˙

3

dropout_13%"

dropout_13˙˙˙˙˙˙˙˙˙
1
	flatten_7$!
	flatten_7˙˙˙˙˙˙˙˙˙
D
flatten_7_input1.
flatten_7_input˙˙˙˙˙˙˙˙˙
-
layer_2"
layer_2˙˙˙˙˙˙˙˙˙Ł
B__inference_dense_13_layer_call_and_return_conditional_losses_1284]0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 {
'__inference_dense_13_layer_call_fn_1274P0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
Ś
D__inference_dropout_13_layer_call_and_return_conditional_losses_1253^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ś
D__inference_dropout_13_layer_call_and_return_conditional_losses_1265^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
)__inference_dropout_13_layer_call_fn_1243Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙~
)__inference_dropout_13_layer_call_fn_1248Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙¨
C__inference_flatten_7_layer_call_and_return_conditional_losses_1206a7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
(__inference_flatten_7_layer_call_fn_1200T7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ł
A__inference_layer_2_layer_call_and_return_conditional_losses_1238^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 {
&__inference_layer_2_layer_call_fn_1221Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙9
__inference_loss_fn_0_1295˘

˘ 
Ş " Ô
?__inference_model_layer_call_and_return_conditional_losses_1031H˘E
>˘;
1.
flatten_7_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "˝˘š
ą­
,)
'$
0/0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙

0/2˙˙˙˙˙˙˙˙˙

0/3˙˙˙˙˙˙˙˙˙

0/4˙˙˙˙˙˙˙˙˙

 Ô
?__inference_model_layer_call_and_return_conditional_losses_1057H˘E
>˘;
1.
flatten_7_input˙˙˙˙˙˙˙˙˙
p

 
Ş "˝˘š
ą­
,)
'$
0/0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙

0/2˙˙˙˙˙˙˙˙˙

0/3˙˙˙˙˙˙˙˙˙

0/4˙˙˙˙˙˙˙˙˙

 Ë
?__inference_model_layer_call_and_return_conditional_losses_1158?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˝˘š
ą­
,)
'$
0/0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙

0/2˙˙˙˙˙˙˙˙˙

0/3˙˙˙˙˙˙˙˙˙

0/4˙˙˙˙˙˙˙˙˙

 Ë
?__inference_model_layer_call_and_return_conditional_losses_1195?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˝˘š
ą­
,)
'$
0/0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙

0/2˙˙˙˙˙˙˙˙˙

0/3˙˙˙˙˙˙˙˙˙

0/4˙˙˙˙˙˙˙˙˙

 Ł
$__inference_model_layer_call_fn_1005úH˘E
>˘;
1.
flatten_7_input˙˙˙˙˙˙˙˙˙
p

 
Ş "§Ł
*'
%"
0/0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙

2˙˙˙˙˙˙˙˙˙

3˙˙˙˙˙˙˙˙˙

4˙˙˙˙˙˙˙˙˙

$__inference_model_layer_call_fn_1107ń?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "§Ł
*'
%"
0/0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙

2˙˙˙˙˙˙˙˙˙

3˙˙˙˙˙˙˙˙˙

4˙˙˙˙˙˙˙˙˙

$__inference_model_layer_call_fn_1128ń?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "§Ł
*'
%"
0/0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙

2˙˙˙˙˙˙˙˙˙

3˙˙˙˙˙˙˙˙˙

4˙˙˙˙˙˙˙˙˙
˘
#__inference_model_layer_call_fn_867úH˘E
>˘;
1.
flatten_7_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "§Ł
*'
%"
0/0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙

2˙˙˙˙˙˙˙˙˙

3˙˙˙˙˙˙˙˙˙

4˙˙˙˙˙˙˙˙˙

"__inference_signature_wrapper_1086ďS˘P
˘ 
IŞF
D
flatten_7_input1.
flatten_7_input˙˙˙˙˙˙˙˙˙"Ş
.
dense_13"
dense_13˙˙˙˙˙˙˙˙˙

3

dropout_13%"

dropout_13˙˙˙˙˙˙˙˙˙
1
	flatten_7$!
	flatten_7˙˙˙˙˙˙˙˙˙
D
flatten_7_input1.
flatten_7_input˙˙˙˙˙˙˙˙˙
-
layer_2"
layer_2˙˙˙˙˙˙˙˙˙