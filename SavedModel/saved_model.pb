??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
?
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
?
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
executor_typestring ??
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
z
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

:*
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_40/kernel
}
$conv2d_40/kernel/Read/ReadVariableOpReadVariableOpconv2d_40/kernel*&
_output_shapes
:*
dtype0
t
conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_40/bias
m
"conv2d_40/bias/Read/ReadVariableOpReadVariableOpconv2d_40/bias*
_output_shapes
:*
dtype0
?
conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_41/kernel
}
$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_41/bias
m
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes
:@*
dtype0
?
conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_42/kernel
}
$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_42/bias
m
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes
:@*
dtype0
?
conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_43/kernel
~
$conv2d_43/kernel/Read/ReadVariableOpReadVariableOpconv2d_43/kernel*'
_output_shapes
:@?*
dtype0
u
conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_43/bias
n
"conv2d_43/bias/Read/ReadVariableOpReadVariableOpconv2d_43/bias*
_output_shapes	
:?*
dtype0
|
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?@? * 
shared_namedense_30/kernel
u
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel* 
_output_shapes
:
?@? *
dtype0
s
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_30/bias
l
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes	
:? *
dtype0
|
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?* 
shared_namedense_31/kernel
u
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel* 
_output_shapes
:
? ?*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_32/kernel/m
?
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/m
y
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_40/kernel/m
?
+Adam/conv2d_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_40/bias/m
{
)Adam/conv2d_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_41/kernel/m
?
+Adam/conv2d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/m*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_41/bias/m
{
)Adam/conv2d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_42/kernel/m
?
+Adam/conv2d_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_42/bias/m
{
)Adam/conv2d_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_43/kernel/m
?
+Adam/conv2d_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_43/bias/m
|
)Adam/conv2d_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?@? *'
shared_nameAdam/dense_30/kernel/m
?
*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m* 
_output_shapes
:
?@? *
dtype0
?
Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *%
shared_nameAdam/dense_30/bias/m
z
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
_output_shapes	
:? *
dtype0
?
Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*'
shared_nameAdam/dense_31/kernel/m
?
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m* 
_output_shapes
:
? ?*
dtype0
?
Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/m
z
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_32/kernel/v
?
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/v
y
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_40/kernel/v
?
+Adam/conv2d_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_40/bias/v
{
)Adam/conv2d_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_41/kernel/v
?
+Adam/conv2d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_41/bias/v
{
)Adam/conv2d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_42/kernel/v
?
+Adam/conv2d_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_42/bias/v
{
)Adam/conv2d_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_43/kernel/v
?
+Adam/conv2d_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_43/bias/v
|
)Adam/conv2d_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?@? *'
shared_nameAdam/dense_30/kernel/v
?
*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v* 
_output_shapes
:
?@? *
dtype0
?
Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *%
shared_nameAdam/dense_30/bias/v
z
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
_output_shapes	
:? *
dtype0
?
Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*'
shared_nameAdam/dense_31/kernel/v
?
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v* 
_output_shapes
:
? ?*
dtype0
?
Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/v
z
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?x
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?x
value?xB?x B?x
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer_with_weights-3
layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
?
0iter

1beta_1

2beta_2
	3decay
4learning_rate(m?)m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?(v?)v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?*
j
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
(12
)13*
j
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
(12
)13*
* 
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Fserving_default* 
?

5kernel
6bias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W_random_generator
X__call__
*Y&call_and_return_all_conditional_losses* 
?

7kernel
8bias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses*
?
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses* 
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j_random_generator
k__call__
*l&call_and_return_all_conditional_losses* 
?

9kernel
:bias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses*
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses* 
?
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}_random_generator
~__call__
*&call_and_return_all_conditional_losses* 
?

;kernel
<bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

=kernel
>bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

?kernel
@bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
Z
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11*
Z
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_32/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_40/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_40/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_41/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_41/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_42/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_42/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_43/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_43/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_30/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_30/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_31/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_31/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

?0
?1*
* 
* 
* 

50
61*

50
61*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 
* 
* 
* 

70
81*

70
81*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 
* 
* 
* 

90
:1*

90
:1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
* 

;0
<1*

;0
<1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

=0
>1*

=0
>1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

?0
@1*

?0
@1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
?|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_40/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_40/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_41/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_41/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_42/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_42/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_43/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_43/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_30/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_30/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_31/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_31/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_40/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_40/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_41/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_41/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_42/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_42/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_43/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_43/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_30/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_30/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_31/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_31/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_anchorPlaceholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
serving_default_comparePlaceholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_anchorserving_default_compareconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_140317
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv2d_40/kernel/Read/ReadVariableOp"conv2d_40/bias/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp$conv2d_42/kernel/Read/ReadVariableOp"conv2d_42/bias/Read/ReadVariableOp$conv2d_43/kernel/Read/ReadVariableOp"conv2d_43/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp+Adam/conv2d_40/kernel/m/Read/ReadVariableOp)Adam/conv2d_40/bias/m/Read/ReadVariableOp+Adam/conv2d_41/kernel/m/Read/ReadVariableOp)Adam/conv2d_41/bias/m/Read/ReadVariableOp+Adam/conv2d_42/kernel/m/Read/ReadVariableOp)Adam/conv2d_42/bias/m/Read/ReadVariableOp+Adam/conv2d_43/kernel/m/Read/ReadVariableOp)Adam/conv2d_43/bias/m/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp+Adam/conv2d_40/kernel/v/Read/ReadVariableOp)Adam/conv2d_40/bias/v/Read/ReadVariableOp+Adam/conv2d_41/kernel/v/Read/ReadVariableOp)Adam/conv2d_41/bias/v/Read/ReadVariableOp+Adam/conv2d_42/kernel/v/Read/ReadVariableOp)Adam/conv2d_42/bias/v/Read/ReadVariableOp+Adam/conv2d_43/kernel/v/Read/ReadVariableOp)Adam/conv2d_43/bias/v/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_140963
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_32/kerneldense_32/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/biastotalcounttotal_1count_1Adam/dense_32/kernel/mAdam/dense_32/bias/mAdam/conv2d_40/kernel/mAdam/conv2d_40/bias/mAdam/conv2d_41/kernel/mAdam/conv2d_41/bias/mAdam/conv2d_42/kernel/mAdam/conv2d_42/bias/mAdam/conv2d_43/kernel/mAdam/conv2d_43/bias/mAdam/dense_30/kernel/mAdam/dense_30/bias/mAdam/dense_31/kernel/mAdam/dense_31/bias/mAdam/dense_32/kernel/vAdam/dense_32/bias/vAdam/conv2d_40/kernel/vAdam/conv2d_40/bias/vAdam/conv2d_41/kernel/vAdam/conv2d_41/bias/vAdam/conv2d_42/kernel/vAdam/conv2d_42/bias/vAdam/conv2d_43/kernel/vAdam/conv2d_43/bias/vAdam/dense_30/kernel/vAdam/dense_30/bias/vAdam/dense_31/kernel/vAdam/dense_31/bias/v*?
Tin8
624*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_141126??
?
d
+__inference_dropout_31_layer_call_fn_140641

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_31_layer_call_and_return_conditional_losses_139313w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_model_10_layer_call_fn_140021
inputs_0
inputs_1!
unknown:
	unknown_0:#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:
?@? 
	unknown_8:	? 
	unknown_9:
? ?

unknown_10:	?

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_10_layer_call_and_return_conditional_losses_139784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
?

v
M__inference_distance_layer_10_layer_call_and_return_conditional_losses_140524

anchor
compare
identityN
subSubanchorcompare*
T0*(
_output_shapes
:??????????L
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????M
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????P
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameanchor:QM
(
_output_shapes
:??????????
!
_user_specified_name	compare
?
?
)__inference_model_10_layer_call_fn_139849

anchor
compare!
unknown:
	unknown_0:#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:
?@? 
	unknown_8:	? 
	unknown_9:
? ?

unknown_10:	?

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallanchorcompareunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_10_layer_call_and_return_conditional_losses_139784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameanchor:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	compare
?
?
*__inference_conv2d_40_layer_call_fn_140553

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_139073w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
b
F__inference_flatten_10_layer_call_and_return_conditional_losses_140746

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_30_layer_call_fn_140579

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_139085h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
E__inference_conv2d_41_layer_call_and_return_conditional_losses_139098

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????  @_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
D__inference_model_10_layer_call_and_return_conditional_losses_139650

inputs
inputs_1.
sequential_10_139578:"
sequential_10_139580:.
sequential_10_139582:@"
sequential_10_139584:@.
sequential_10_139586:@@"
sequential_10_139588:@/
sequential_10_139590:@?#
sequential_10_139592:	?(
sequential_10_139594:
?@? #
sequential_10_139596:	? (
sequential_10_139598:
? ?#
sequential_10_139600:	?!
dense_32_139644:
dense_32_139646:
identity?? dense_32/StatefulPartitionedCall?%sequential_10/StatefulPartitionedCall?'sequential_10/StatefulPartitionedCall_1?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinputssequential_10_139578sequential_10_139580sequential_10_139582sequential_10_139584sequential_10_139586sequential_10_139588sequential_10_139590sequential_10_139592sequential_10_139594sequential_10_139596sequential_10_139598sequential_10_139600*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139197?
'sequential_10/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_10_139578sequential_10_139580sequential_10_139582sequential_10_139584sequential_10_139586sequential_10_139588sequential_10_139590sequential_10_139592sequential_10_139594sequential_10_139596sequential_10_139598sequential_10_139600*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139197?
!distance_layer_10/PartitionedCallPartitionedCall.sequential_10/StatefulPartitionedCall:output:00sequential_10/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_distance_layer_10_layer_call_and_return_conditional_losses_139630?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall*distance_layer_10/PartitionedCall:output:0dense_32_139644dense_32_139646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_139643x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_32/StatefulPartitionedCall&^sequential_10/StatefulPartitionedCall(^sequential_10/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall2R
'sequential_10/StatefulPartitionedCall_1'sequential_10/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
.__inference_sequential_10_layer_call_fn_140346

inputs!
unknown:
	unknown_0:#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:
?@? 
	unknown_8:	? 
	unknown_9:
? ?

unknown_10:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139197p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_32_layer_call_fn_140683

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_139052?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_30_layer_call_and_return_conditional_losses_139085

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?5
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_139197

inputs*
conv2d_40_139074:
conv2d_40_139076:*
conv2d_41_139099:@
conv2d_41_139101:@*
conv2d_42_139124:@@
conv2d_42_139126:@+
conv2d_43_139149:@?
conv2d_43_139151:	?#
dense_30_139174:
?@? 
dense_30_139176:	? #
dense_31_139191:
? ?
dense_31_139193:	?
identity??!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_139074conv2d_40_139076*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_139073?
 max_pooling2d_30/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_139028?
dropout_30/PartitionedCallPartitionedCall)max_pooling2d_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_139085?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_41_139099conv2d_41_139101*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_139098?
 max_pooling2d_31/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_139040?
dropout_31/PartitionedCallPartitionedCall)max_pooling2d_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_31_layer_call_and_return_conditional_losses_139110?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0conv2d_42_139124conv2d_42_139126*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_139123?
 max_pooling2d_32/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_139052?
dropout_32/PartitionedCallPartitionedCall)max_pooling2d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_139135?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_32/PartitionedCall:output:0conv2d_43_139149conv2d_43_139151*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_139148?
flatten_10/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_139160?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_30_139174dense_30_139176*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_139173?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_139191dense_31_139193*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_139190y
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_40_layer_call_and_return_conditional_losses_139073

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
D__inference_dense_32_layer_call_and_return_conditional_losses_139643

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_32_layer_call_fn_140533

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_139643o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_141126
file_prefix2
 assignvariableop_dense_32_kernel:.
 assignvariableop_1_dense_32_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: =
#assignvariableop_7_conv2d_40_kernel:/
!assignvariableop_8_conv2d_40_bias:=
#assignvariableop_9_conv2d_41_kernel:@0
"assignvariableop_10_conv2d_41_bias:@>
$assignvariableop_11_conv2d_42_kernel:@@0
"assignvariableop_12_conv2d_42_bias:@?
$assignvariableop_13_conv2d_43_kernel:@?1
"assignvariableop_14_conv2d_43_bias:	?7
#assignvariableop_15_dense_30_kernel:
?@? 0
!assignvariableop_16_dense_30_bias:	? 7
#assignvariableop_17_dense_31_kernel:
? ?0
!assignvariableop_18_dense_31_bias:	?#
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: <
*assignvariableop_23_adam_dense_32_kernel_m:6
(assignvariableop_24_adam_dense_32_bias_m:E
+assignvariableop_25_adam_conv2d_40_kernel_m:7
)assignvariableop_26_adam_conv2d_40_bias_m:E
+assignvariableop_27_adam_conv2d_41_kernel_m:@7
)assignvariableop_28_adam_conv2d_41_bias_m:@E
+assignvariableop_29_adam_conv2d_42_kernel_m:@@7
)assignvariableop_30_adam_conv2d_42_bias_m:@F
+assignvariableop_31_adam_conv2d_43_kernel_m:@?8
)assignvariableop_32_adam_conv2d_43_bias_m:	?>
*assignvariableop_33_adam_dense_30_kernel_m:
?@? 7
(assignvariableop_34_adam_dense_30_bias_m:	? >
*assignvariableop_35_adam_dense_31_kernel_m:
? ?7
(assignvariableop_36_adam_dense_31_bias_m:	?<
*assignvariableop_37_adam_dense_32_kernel_v:6
(assignvariableop_38_adam_dense_32_bias_v:E
+assignvariableop_39_adam_conv2d_40_kernel_v:7
)assignvariableop_40_adam_conv2d_40_bias_v:E
+assignvariableop_41_adam_conv2d_41_kernel_v:@7
)assignvariableop_42_adam_conv2d_41_bias_v:@E
+assignvariableop_43_adam_conv2d_42_kernel_v:@@7
)assignvariableop_44_adam_conv2d_42_bias_v:@F
+assignvariableop_45_adam_conv2d_43_kernel_v:@?8
)assignvariableop_46_adam_conv2d_43_bias_v:	?>
*assignvariableop_47_adam_dense_30_kernel_v:
?@? 7
(assignvariableop_48_adam_dense_30_bias_v:	? >
*assignvariableop_49_adam_dense_31_kernel_v:
? ?7
(assignvariableop_50_adam_dense_31_bias_v:	?
identity_52??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_32_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_32_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_40_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_40_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_41_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_41_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_42_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_42_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_43_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_43_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_30_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_30_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_31_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_31_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_32_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_32_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_40_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_40_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_41_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_41_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_42_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_42_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_43_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_43_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_30_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_30_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_31_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_31_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_32_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_32_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_40_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_40_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_41_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_41_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_42_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_42_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_43_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_43_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_30_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_30_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_31_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_31_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: ?	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_52Identity_52:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
d
F__inference_dropout_31_layer_call_and_return_conditional_losses_140646

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_dense_31_layer_call_and_return_conditional_losses_139190

inputs2
matmul_readvariableop_resource:
? ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_139052

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_model_10_layer_call_and_return_conditional_losses_139784

inputs
inputs_1.
sequential_10_139739:"
sequential_10_139741:.
sequential_10_139743:@"
sequential_10_139745:@.
sequential_10_139747:@@"
sequential_10_139749:@/
sequential_10_139751:@?#
sequential_10_139753:	?(
sequential_10_139755:
?@? #
sequential_10_139757:	? (
sequential_10_139759:
? ?#
sequential_10_139761:	?!
dense_32_139778:
dense_32_139780:
identity?? dense_32/StatefulPartitionedCall?%sequential_10/StatefulPartitionedCall?'sequential_10/StatefulPartitionedCall_1?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinputssequential_10_139739sequential_10_139741sequential_10_139743sequential_10_139745sequential_10_139747sequential_10_139749sequential_10_139751sequential_10_139753sequential_10_139755sequential_10_139757sequential_10_139759sequential_10_139761*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139431?
'sequential_10/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_10_139739sequential_10_139741sequential_10_139743sequential_10_139745sequential_10_139747sequential_10_139749sequential_10_139751sequential_10_139753sequential_10_139755sequential_10_139757sequential_10_139759sequential_10_139761*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139431?
!distance_layer_10/PartitionedCallPartitionedCall.sequential_10/StatefulPartitionedCall:output:00sequential_10/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_distance_layer_10_layer_call_and_return_conditional_losses_139630?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall*distance_layer_10/PartitionedCall:output:0dense_32_139778dense_32_139780*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_139643x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_32/StatefulPartitionedCall&^sequential_10/StatefulPartitionedCall(^sequential_10/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall2R
'sequential_10/StatefulPartitionedCall_1'sequential_10/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_139028

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_32_layer_call_fn_140693

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_139135h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
.__inference_sequential_10_layer_call_fn_139224
conv2d_40_input!
unknown:
	unknown_0:#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:
?@? 
	unknown_8:	? 
	unknown_9:
? ?

unknown_10:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139197p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_40_input
?
?
E__inference_conv2d_43_layer_call_and_return_conditional_losses_140735

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
TanhTanhBiasAdd:output:0*
T0*0
_output_shapes
:??????????`
IdentityIdentityTanh:y:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_model_10_layer_call_and_return_conditional_losses_139947

anchor
compare.
sequential_10_139902:"
sequential_10_139904:.
sequential_10_139906:@"
sequential_10_139908:@.
sequential_10_139910:@@"
sequential_10_139912:@/
sequential_10_139914:@?#
sequential_10_139916:	?(
sequential_10_139918:
?@? #
sequential_10_139920:	? (
sequential_10_139922:
? ?#
sequential_10_139924:	?!
dense_32_139941:
dense_32_139943:
identity?? dense_32/StatefulPartitionedCall?%sequential_10/StatefulPartitionedCall?'sequential_10/StatefulPartitionedCall_1?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallanchorsequential_10_139902sequential_10_139904sequential_10_139906sequential_10_139908sequential_10_139910sequential_10_139912sequential_10_139914sequential_10_139916sequential_10_139918sequential_10_139920sequential_10_139922sequential_10_139924*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139431?
'sequential_10/StatefulPartitionedCall_1StatefulPartitionedCallcomparesequential_10_139902sequential_10_139904sequential_10_139906sequential_10_139908sequential_10_139910sequential_10_139912sequential_10_139914sequential_10_139916sequential_10_139918sequential_10_139920sequential_10_139922sequential_10_139924*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139431?
!distance_layer_10/PartitionedCallPartitionedCall.sequential_10/StatefulPartitionedCall:output:00sequential_10/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_distance_layer_10_layer_call_and_return_conditional_losses_139630?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall*distance_layer_10/PartitionedCall:output:0dense_32_139941dense_32_139943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_139643x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_32/StatefulPartitionedCall&^sequential_10/StatefulPartitionedCall(^sequential_10/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall2R
'sequential_10/StatefulPartitionedCall_1'sequential_10/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameanchor:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	compare
?
d
+__inference_dropout_32_layer_call_fn_140698

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_139280w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
ݫ
?
D__inference_model_10_layer_call_and_return_conditional_losses_140130
inputs_0
inputs_1P
6sequential_10_conv2d_40_conv2d_readvariableop_resource:E
7sequential_10_conv2d_40_biasadd_readvariableop_resource:P
6sequential_10_conv2d_41_conv2d_readvariableop_resource:@E
7sequential_10_conv2d_41_biasadd_readvariableop_resource:@P
6sequential_10_conv2d_42_conv2d_readvariableop_resource:@@E
7sequential_10_conv2d_42_biasadd_readvariableop_resource:@Q
6sequential_10_conv2d_43_conv2d_readvariableop_resource:@?F
7sequential_10_conv2d_43_biasadd_readvariableop_resource:	?I
5sequential_10_dense_30_matmul_readvariableop_resource:
?@? E
6sequential_10_dense_30_biasadd_readvariableop_resource:	? I
5sequential_10_dense_31_matmul_readvariableop_resource:
? ?E
6sequential_10_dense_31_biasadd_readvariableop_resource:	?9
'dense_32_matmul_readvariableop_resource:6
(dense_32_biasadd_readvariableop_resource:
identity??dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?.sequential_10/conv2d_40/BiasAdd/ReadVariableOp?0sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp?-sequential_10/conv2d_40/Conv2D/ReadVariableOp?/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp?.sequential_10/conv2d_41/BiasAdd/ReadVariableOp?0sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp?-sequential_10/conv2d_41/Conv2D/ReadVariableOp?/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp?.sequential_10/conv2d_42/BiasAdd/ReadVariableOp?0sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp?-sequential_10/conv2d_42/Conv2D/ReadVariableOp?/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp?.sequential_10/conv2d_43/BiasAdd/ReadVariableOp?0sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp?-sequential_10/conv2d_43/Conv2D/ReadVariableOp?/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp?-sequential_10/dense_30/BiasAdd/ReadVariableOp?/sequential_10/dense_30/BiasAdd_1/ReadVariableOp?,sequential_10/dense_30/MatMul/ReadVariableOp?.sequential_10/dense_30/MatMul_1/ReadVariableOp?-sequential_10/dense_31/BiasAdd/ReadVariableOp?/sequential_10/dense_31/BiasAdd_1/ReadVariableOp?,sequential_10/dense_31/MatMul/ReadVariableOp?.sequential_10/dense_31/MatMul_1/ReadVariableOp?
-sequential_10/conv2d_40/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_10/conv2d_40/Conv2DConv2Dinputs_05sequential_10/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
.sequential_10/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_10/conv2d_40/BiasAddBiasAdd'sequential_10/conv2d_40/Conv2D:output:06sequential_10/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
sequential_10/conv2d_40/TanhTanh(sequential_10/conv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
&sequential_10/max_pooling2d_30/MaxPoolMaxPool sequential_10/conv2d_40/Tanh:y:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
?
!sequential_10/dropout_30/IdentityIdentity/sequential_10/max_pooling2d_30/MaxPool:output:0*
T0*/
_output_shapes
:?????????  ?
-sequential_10/conv2d_41/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
sequential_10/conv2d_41/Conv2DConv2D*sequential_10/dropout_30/Identity:output:05sequential_10/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
.sequential_10/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_10/conv2d_41/BiasAddBiasAdd'sequential_10/conv2d_41/Conv2D:output:06sequential_10/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
sequential_10/conv2d_41/TanhTanh(sequential_10/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @?
&sequential_10/max_pooling2d_31/MaxPoolMaxPool sequential_10/conv2d_41/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
!sequential_10/dropout_31/IdentityIdentity/sequential_10/max_pooling2d_31/MaxPool:output:0*
T0*/
_output_shapes
:?????????@?
-sequential_10/conv2d_42/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
sequential_10/conv2d_42/Conv2DConv2D*sequential_10/dropout_31/Identity:output:05sequential_10/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
.sequential_10/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_10/conv2d_42/BiasAddBiasAdd'sequential_10/conv2d_42/Conv2D:output:06sequential_10/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
sequential_10/conv2d_42/TanhTanh(sequential_10/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
&sequential_10/max_pooling2d_32/MaxPoolMaxPool sequential_10/conv2d_42/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
!sequential_10/dropout_32/IdentityIdentity/sequential_10/max_pooling2d_32/MaxPool:output:0*
T0*/
_output_shapes
:?????????@?
-sequential_10/conv2d_43/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential_10/conv2d_43/Conv2DConv2D*sequential_10/dropout_32/Identity:output:05sequential_10/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
.sequential_10/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_10/conv2d_43/BiasAddBiasAdd'sequential_10/conv2d_43/Conv2D:output:06sequential_10/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
sequential_10/conv2d_43/TanhTanh(sequential_10/conv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:??????????o
sequential_10/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
 sequential_10/flatten_10/ReshapeReshape sequential_10/conv2d_43/Tanh:y:0'sequential_10/flatten_10/Const:output:0*
T0*(
_output_shapes
:??????????@?
,sequential_10/dense_30/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0?
sequential_10/dense_30/MatMulMatMul)sequential_10/flatten_10/Reshape:output:04sequential_10/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
-sequential_10/dense_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
sequential_10/dense_30/BiasAddBiasAdd'sequential_10/dense_30/MatMul:product:05sequential_10/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 
sequential_10/dense_30/TanhTanh'sequential_10/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? ?
,sequential_10/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0?
sequential_10/dense_31/MatMulMatMulsequential_10/dense_30/Tanh:y:04sequential_10/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_10/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_10/dense_31/BiasAddBiasAdd'sequential_10/dense_31/MatMul:product:05sequential_10/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential_10/dense_31/SigmoidSigmoid'sequential_10/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
/sequential_10/conv2d_40/Conv2D_1/ReadVariableOpReadVariableOp6sequential_10_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
 sequential_10/conv2d_40/Conv2D_1Conv2Dinputs_17sequential_10/conv2d_40/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
0sequential_10/conv2d_40/BiasAdd_1/ReadVariableOpReadVariableOp7sequential_10_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
!sequential_10/conv2d_40/BiasAdd_1BiasAdd)sequential_10/conv2d_40/Conv2D_1:output:08sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
sequential_10/conv2d_40/Tanh_1Tanh*sequential_10/conv2d_40/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????@@?
(sequential_10/max_pooling2d_30/MaxPool_1MaxPool"sequential_10/conv2d_40/Tanh_1:y:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
?
#sequential_10/dropout_30/Identity_1Identity1sequential_10/max_pooling2d_30/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????  ?
/sequential_10/conv2d_41/Conv2D_1/ReadVariableOpReadVariableOp6sequential_10_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
 sequential_10/conv2d_41/Conv2D_1Conv2D,sequential_10/dropout_30/Identity_1:output:07sequential_10/conv2d_41/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
0sequential_10/conv2d_41/BiasAdd_1/ReadVariableOpReadVariableOp7sequential_10_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
!sequential_10/conv2d_41/BiasAdd_1BiasAdd)sequential_10/conv2d_41/Conv2D_1:output:08sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
sequential_10/conv2d_41/Tanh_1Tanh*sequential_10/conv2d_41/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????  @?
(sequential_10/max_pooling2d_31/MaxPool_1MaxPool"sequential_10/conv2d_41/Tanh_1:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
#sequential_10/dropout_31/Identity_1Identity1sequential_10/max_pooling2d_31/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????@?
/sequential_10/conv2d_42/Conv2D_1/ReadVariableOpReadVariableOp6sequential_10_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
 sequential_10/conv2d_42/Conv2D_1Conv2D,sequential_10/dropout_31/Identity_1:output:07sequential_10/conv2d_42/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
0sequential_10/conv2d_42/BiasAdd_1/ReadVariableOpReadVariableOp7sequential_10_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
!sequential_10/conv2d_42/BiasAdd_1BiasAdd)sequential_10/conv2d_42/Conv2D_1:output:08sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
sequential_10/conv2d_42/Tanh_1Tanh*sequential_10/conv2d_42/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????@?
(sequential_10/max_pooling2d_32/MaxPool_1MaxPool"sequential_10/conv2d_42/Tanh_1:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
#sequential_10/dropout_32/Identity_1Identity1sequential_10/max_pooling2d_32/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????@?
/sequential_10/conv2d_43/Conv2D_1/ReadVariableOpReadVariableOp6sequential_10_conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
 sequential_10/conv2d_43/Conv2D_1Conv2D,sequential_10/dropout_32/Identity_1:output:07sequential_10/conv2d_43/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
0sequential_10/conv2d_43/BiasAdd_1/ReadVariableOpReadVariableOp7sequential_10_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!sequential_10/conv2d_43/BiasAdd_1BiasAdd)sequential_10/conv2d_43/Conv2D_1:output:08sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
sequential_10/conv2d_43/Tanh_1Tanh*sequential_10/conv2d_43/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????q
 sequential_10/flatten_10/Const_1Const*
_output_shapes
:*
dtype0*
valueB"????    ?
"sequential_10/flatten_10/Reshape_1Reshape"sequential_10/conv2d_43/Tanh_1:y:0)sequential_10/flatten_10/Const_1:output:0*
T0*(
_output_shapes
:??????????@?
.sequential_10/dense_30/MatMul_1/ReadVariableOpReadVariableOp5sequential_10_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0?
sequential_10/dense_30/MatMul_1MatMul+sequential_10/flatten_10/Reshape_1:output:06sequential_10/dense_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
/sequential_10/dense_30/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_10_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
 sequential_10/dense_30/BiasAdd_1BiasAdd)sequential_10/dense_30/MatMul_1:product:07sequential_10/dense_30/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
sequential_10/dense_30/Tanh_1Tanh)sequential_10/dense_30/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????? ?
.sequential_10/dense_31/MatMul_1/ReadVariableOpReadVariableOp5sequential_10_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0?
sequential_10/dense_31/MatMul_1MatMul!sequential_10/dense_30/Tanh_1:y:06sequential_10/dense_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
/sequential_10/dense_31/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_10_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
 sequential_10/dense_31/BiasAdd_1BiasAdd)sequential_10/dense_31/MatMul_1:product:07sequential_10/dense_31/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 sequential_10/dense_31/Sigmoid_1Sigmoid)sequential_10/dense_31/BiasAdd_1:output:0*
T0*(
_output_shapes
:???????????
distance_layer_10/subSub"sequential_10/dense_31/Sigmoid:y:0$sequential_10/dense_31/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????p
distance_layer_10/SquareSquaredistance_layer_10/sub:z:0*
T0*(
_output_shapes
:??????????i
'distance_layer_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
distance_layer_10/SumSumdistance_layer_10/Square:y:00distance_layer_10/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(`
distance_layer_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
distance_layer_10/MaximumMaximumdistance_layer_10/Sum:output:0$distance_layer_10/Maximum/y:output:0*
T0*'
_output_shapes
:?????????\
distance_layer_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
distance_layer_10/Maximum_1Maximumdistance_layer_10/Maximum:z:0 distance_layer_10/Const:output:0*
T0*'
_output_shapes
:?????????q
distance_layer_10/SqrtSqrtdistance_layer_10/Maximum_1:z:0*
T0*'
_output_shapes
:??????????
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_32/MatMulMatMuldistance_layer_10/Sqrt:y:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_32/SigmoidSigmoiddense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_32/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????

NoOpNoOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp/^sequential_10/conv2d_40/BiasAdd/ReadVariableOp1^sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp.^sequential_10/conv2d_40/Conv2D/ReadVariableOp0^sequential_10/conv2d_40/Conv2D_1/ReadVariableOp/^sequential_10/conv2d_41/BiasAdd/ReadVariableOp1^sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp.^sequential_10/conv2d_41/Conv2D/ReadVariableOp0^sequential_10/conv2d_41/Conv2D_1/ReadVariableOp/^sequential_10/conv2d_42/BiasAdd/ReadVariableOp1^sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp.^sequential_10/conv2d_42/Conv2D/ReadVariableOp0^sequential_10/conv2d_42/Conv2D_1/ReadVariableOp/^sequential_10/conv2d_43/BiasAdd/ReadVariableOp1^sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp.^sequential_10/conv2d_43/Conv2D/ReadVariableOp0^sequential_10/conv2d_43/Conv2D_1/ReadVariableOp.^sequential_10/dense_30/BiasAdd/ReadVariableOp0^sequential_10/dense_30/BiasAdd_1/ReadVariableOp-^sequential_10/dense_30/MatMul/ReadVariableOp/^sequential_10/dense_30/MatMul_1/ReadVariableOp.^sequential_10/dense_31/BiasAdd/ReadVariableOp0^sequential_10/dense_31/BiasAdd_1/ReadVariableOp-^sequential_10/dense_31/MatMul/ReadVariableOp/^sequential_10/dense_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2`
.sequential_10/conv2d_40/BiasAdd/ReadVariableOp.sequential_10/conv2d_40/BiasAdd/ReadVariableOp2d
0sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp0sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp2^
-sequential_10/conv2d_40/Conv2D/ReadVariableOp-sequential_10/conv2d_40/Conv2D/ReadVariableOp2b
/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp2`
.sequential_10/conv2d_41/BiasAdd/ReadVariableOp.sequential_10/conv2d_41/BiasAdd/ReadVariableOp2d
0sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp0sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp2^
-sequential_10/conv2d_41/Conv2D/ReadVariableOp-sequential_10/conv2d_41/Conv2D/ReadVariableOp2b
/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp2`
.sequential_10/conv2d_42/BiasAdd/ReadVariableOp.sequential_10/conv2d_42/BiasAdd/ReadVariableOp2d
0sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp0sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp2^
-sequential_10/conv2d_42/Conv2D/ReadVariableOp-sequential_10/conv2d_42/Conv2D/ReadVariableOp2b
/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp2`
.sequential_10/conv2d_43/BiasAdd/ReadVariableOp.sequential_10/conv2d_43/BiasAdd/ReadVariableOp2d
0sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp0sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp2^
-sequential_10/conv2d_43/Conv2D/ReadVariableOp-sequential_10/conv2d_43/Conv2D/ReadVariableOp2b
/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp2^
-sequential_10/dense_30/BiasAdd/ReadVariableOp-sequential_10/dense_30/BiasAdd/ReadVariableOp2b
/sequential_10/dense_30/BiasAdd_1/ReadVariableOp/sequential_10/dense_30/BiasAdd_1/ReadVariableOp2\
,sequential_10/dense_30/MatMul/ReadVariableOp,sequential_10/dense_30/MatMul/ReadVariableOp2`
.sequential_10/dense_30/MatMul_1/ReadVariableOp.sequential_10/dense_30/MatMul_1/ReadVariableOp2^
-sequential_10/dense_31/BiasAdd/ReadVariableOp-sequential_10/dense_31/BiasAdd/ReadVariableOp2b
/sequential_10/dense_31/BiasAdd_1/ReadVariableOp/sequential_10/dense_31/BiasAdd_1/ReadVariableOp2\
,sequential_10/dense_31/MatMul/ReadVariableOp,sequential_10/dense_31/MatMul/ReadVariableOp2`
.sequential_10/dense_31/MatMul_1/ReadVariableOp.sequential_10/dense_31/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
?

e
F__inference_dropout_31_layer_call_and_return_conditional_losses_139313

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?A
?	
I__inference_sequential_10_layer_call_and_return_conditional_losses_140429

inputsB
(conv2d_40_conv2d_readvariableop_resource:7
)conv2d_40_biasadd_readvariableop_resource:B
(conv2d_41_conv2d_readvariableop_resource:@7
)conv2d_41_biasadd_readvariableop_resource:@B
(conv2d_42_conv2d_readvariableop_resource:@@7
)conv2d_42_biasadd_readvariableop_resource:@C
(conv2d_43_conv2d_readvariableop_resource:@?8
)conv2d_43_biasadd_readvariableop_resource:	?;
'dense_30_matmul_readvariableop_resource:
?@? 7
(dense_30_biasadd_readvariableop_resource:	? ;
'dense_31_matmul_readvariableop_resource:
? ?7
(dense_31_biasadd_readvariableop_resource:	?
identity?? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@l
conv2d_40/TanhTanhconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
max_pooling2d_30/MaxPoolMaxPoolconv2d_40/Tanh:y:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
|
dropout_30/IdentityIdentity!max_pooling2d_30/MaxPool:output:0*
T0*/
_output_shapes
:?????????  ?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_41/Conv2DConv2Ddropout_30/Identity:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @l
conv2d_41/TanhTanhconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @?
max_pooling2d_31/MaxPoolMaxPoolconv2d_41/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
|
dropout_31/IdentityIdentity!max_pooling2d_31/MaxPool:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_42/Conv2DConv2Ddropout_31/Identity:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_42/TanhTanhconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_32/MaxPoolMaxPoolconv2d_42/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
|
dropout_32/IdentityIdentity!max_pooling2d_32/MaxPool:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_43/Conv2DConv2Ddropout_32/Identity:output:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_43/TanhTanhconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:??????????a
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    
flatten_10/ReshapeReshapeconv2d_43/Tanh:y:0flatten_10/Const:output:0*
T0*(
_output_shapes
:??????????@?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0?
dense_30/MatMulMatMulflatten_10/Reshape:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? c
dense_30/TanhTanhdense_30/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? ?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0?
dense_31/MatMulMatMuldense_30/Tanh:y:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????i
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*(
_output_shapes
:??????????d
IdentityIdentitydense_31/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
D__inference_dense_30_layer_call_and_return_conditional_losses_139173

inputs2
matmul_readvariableop_resource:
?@? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:?????????? X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
*__inference_conv2d_42_layer_call_fn_140667

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_139123w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_model_10_layer_call_and_return_conditional_losses_139898

anchor
compare.
sequential_10_139853:"
sequential_10_139855:.
sequential_10_139857:@"
sequential_10_139859:@.
sequential_10_139861:@@"
sequential_10_139863:@/
sequential_10_139865:@?#
sequential_10_139867:	?(
sequential_10_139869:
?@? #
sequential_10_139871:	? (
sequential_10_139873:
? ?#
sequential_10_139875:	?!
dense_32_139892:
dense_32_139894:
identity?? dense_32/StatefulPartitionedCall?%sequential_10/StatefulPartitionedCall?'sequential_10/StatefulPartitionedCall_1?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallanchorsequential_10_139853sequential_10_139855sequential_10_139857sequential_10_139859sequential_10_139861sequential_10_139863sequential_10_139865sequential_10_139867sequential_10_139869sequential_10_139871sequential_10_139873sequential_10_139875*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139197?
'sequential_10/StatefulPartitionedCall_1StatefulPartitionedCallcomparesequential_10_139853sequential_10_139855sequential_10_139857sequential_10_139859sequential_10_139861sequential_10_139863sequential_10_139865sequential_10_139867sequential_10_139869sequential_10_139871sequential_10_139873sequential_10_139875*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139197?
!distance_layer_10/PartitionedCallPartitionedCall.sequential_10/StatefulPartitionedCall:output:00sequential_10/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_distance_layer_10_layer_call_and_return_conditional_losses_139630?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall*distance_layer_10/PartitionedCall:output:0dense_32_139892dense_32_139894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_139643x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_32/StatefulPartitionedCall&^sequential_10/StatefulPartitionedCall(^sequential_10/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall2R
'sequential_10/StatefulPartitionedCall_1'sequential_10/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameanchor:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	compare
?
G
+__inference_flatten_10_layer_call_fn_140740

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_139160a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_30_layer_call_and_return_conditional_losses_140766

inputs2
matmul_readvariableop_resource:
?@? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:?????????? X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
d
F__inference_dropout_32_layer_call_and_return_conditional_losses_140703

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_dense_30_layer_call_fn_140755

inputs
unknown:
?@? 
	unknown_0:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_139173p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?

e
F__inference_dropout_31_layer_call_and_return_conditional_losses_140658

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
.__inference_sequential_10_layer_call_fn_140375

inputs!
unknown:
	unknown_0:#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:
?@? 
	unknown_8:	? 
	unknown_9:
? ?

unknown_10:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139431p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_model_10_layer_call_fn_139987
inputs_0
inputs_1!
unknown:
	unknown_0:#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:
?@? 
	unknown_8:	? 
	unknown_9:
? ?

unknown_10:	?

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_10_layer_call_and_return_conditional_losses_139650o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
?

e
F__inference_dropout_30_layer_call_and_return_conditional_losses_139346

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????  C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?:
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_139431

inputs*
conv2d_40_139393:
conv2d_40_139395:*
conv2d_41_139400:@
conv2d_41_139402:@*
conv2d_42_139407:@@
conv2d_42_139409:@+
conv2d_43_139414:@?
conv2d_43_139416:	?#
dense_30_139420:
?@? 
dense_30_139422:	? #
dense_31_139425:
? ?
dense_31_139427:	?
identity??!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?"dropout_30/StatefulPartitionedCall?"dropout_31/StatefulPartitionedCall?"dropout_32/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_139393conv2d_40_139395*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_139073?
 max_pooling2d_30/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_139028?
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_139346?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0conv2d_41_139400conv2d_41_139402*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_139098?
 max_pooling2d_31/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_139040?
"dropout_31/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_31/PartitionedCall:output:0#^dropout_30/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_31_layer_call_and_return_conditional_losses_139313?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall+dropout_31/StatefulPartitionedCall:output:0conv2d_42_139407conv2d_42_139409*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_139123?
 max_pooling2d_32/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_139052?
"dropout_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0#^dropout_31/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_139280?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_32/StatefulPartitionedCall:output:0conv2d_43_139414conv2d_43_139416*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_139148?
flatten_10/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_139160?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_30_139420dense_30_139422*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_139173?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_139425dense_31_139427*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_139190y
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall#^dropout_31/StatefulPartitionedCall#^dropout_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall2H
"dropout_31/StatefulPartitionedCall"dropout_31/StatefulPartitionedCall2H
"dropout_32/StatefulPartitionedCall"dropout_32/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_40_layer_call_and_return_conditional_losses_140564

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?c
?
__inference__traced_save_140963
file_prefix.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_conv2d_40_kernel_read_readvariableop-
)savev2_conv2d_40_bias_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop/
+savev2_conv2d_42_kernel_read_readvariableop-
)savev2_conv2d_42_bias_read_readvariableop/
+savev2_conv2d_43_kernel_read_readvariableop-
)savev2_conv2d_43_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableop6
2savev2_adam_conv2d_40_kernel_m_read_readvariableop4
0savev2_adam_conv2d_40_bias_m_read_readvariableop6
2savev2_adam_conv2d_41_kernel_m_read_readvariableop4
0savev2_adam_conv2d_41_bias_m_read_readvariableop6
2savev2_adam_conv2d_42_kernel_m_read_readvariableop4
0savev2_adam_conv2d_42_bias_m_read_readvariableop6
2savev2_adam_conv2d_43_kernel_m_read_readvariableop4
0savev2_adam_conv2d_43_bias_m_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableop6
2savev2_adam_conv2d_40_kernel_v_read_readvariableop4
0savev2_adam_conv2d_40_bias_v_read_readvariableop6
2savev2_adam_conv2d_41_kernel_v_read_readvariableop4
0savev2_adam_conv2d_41_bias_v_read_readvariableop6
2savev2_adam_conv2d_42_kernel_v_read_readvariableop4
0savev2_adam_conv2d_42_bias_v_read_readvariableop6
2savev2_adam_conv2d_43_kernel_v_read_readvariableop4
0savev2_adam_conv2d_43_bias_v_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv2d_40_kernel_read_readvariableop)savev2_conv2d_40_bias_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop+savev2_conv2d_42_kernel_read_readvariableop)savev2_conv2d_42_bias_read_readvariableop+savev2_conv2d_43_kernel_read_readvariableop)savev2_conv2d_43_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop2savev2_adam_conv2d_40_kernel_m_read_readvariableop0savev2_adam_conv2d_40_bias_m_read_readvariableop2savev2_adam_conv2d_41_kernel_m_read_readvariableop0savev2_adam_conv2d_41_bias_m_read_readvariableop2savev2_adam_conv2d_42_kernel_m_read_readvariableop0savev2_adam_conv2d_42_bias_m_read_readvariableop2savev2_adam_conv2d_43_kernel_m_read_readvariableop0savev2_adam_conv2d_43_bias_m_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop2savev2_adam_conv2d_40_kernel_v_read_readvariableop0savev2_adam_conv2d_40_bias_v_read_readvariableop2savev2_adam_conv2d_41_kernel_v_read_readvariableop0savev2_adam_conv2d_41_bias_v_read_readvariableop2savev2_adam_conv2d_42_kernel_v_read_readvariableop0savev2_adam_conv2d_42_bias_v_read_readvariableop2savev2_adam_conv2d_43_kernel_v_read_readvariableop0savev2_adam_conv2d_43_bias_v_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : : : :::@:@:@@:@:@?:?:
?@? :? :
? ?:?: : : : :::::@:@:@@:@:@?:?:
?@? :? :
? ?:?:::::@:@:@@:@:@?:?:
?@? :? :
? ?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
?@? :!

_output_shapes	
:? :&"
 
_output_shapes
:
? ?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:- )
'
_output_shapes
:@?:!!

_output_shapes	
:?:&""
 
_output_shapes
:
?@? :!#

_output_shapes	
:? :&$"
 
_output_shapes
:
? ?:!%

_output_shapes	
:?:$& 

_output_shapes

:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:@: +

_output_shapes
:@:,,(
&
_output_shapes
:@@: -

_output_shapes
:@:-.)
'
_output_shapes
:@?:!/

_output_shapes	
:?:&0"
 
_output_shapes
:
?@? :!1

_output_shapes	
:? :&2"
 
_output_shapes
:
? ?:!3

_output_shapes	
:?:4

_output_shapes
: 
??
?
D__inference_model_10_layer_call_and_return_conditional_losses_140281
inputs_0
inputs_1P
6sequential_10_conv2d_40_conv2d_readvariableop_resource:E
7sequential_10_conv2d_40_biasadd_readvariableop_resource:P
6sequential_10_conv2d_41_conv2d_readvariableop_resource:@E
7sequential_10_conv2d_41_biasadd_readvariableop_resource:@P
6sequential_10_conv2d_42_conv2d_readvariableop_resource:@@E
7sequential_10_conv2d_42_biasadd_readvariableop_resource:@Q
6sequential_10_conv2d_43_conv2d_readvariableop_resource:@?F
7sequential_10_conv2d_43_biasadd_readvariableop_resource:	?I
5sequential_10_dense_30_matmul_readvariableop_resource:
?@? E
6sequential_10_dense_30_biasadd_readvariableop_resource:	? I
5sequential_10_dense_31_matmul_readvariableop_resource:
? ?E
6sequential_10_dense_31_biasadd_readvariableop_resource:	?9
'dense_32_matmul_readvariableop_resource:6
(dense_32_biasadd_readvariableop_resource:
identity??dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?.sequential_10/conv2d_40/BiasAdd/ReadVariableOp?0sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp?-sequential_10/conv2d_40/Conv2D/ReadVariableOp?/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp?.sequential_10/conv2d_41/BiasAdd/ReadVariableOp?0sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp?-sequential_10/conv2d_41/Conv2D/ReadVariableOp?/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp?.sequential_10/conv2d_42/BiasAdd/ReadVariableOp?0sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp?-sequential_10/conv2d_42/Conv2D/ReadVariableOp?/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp?.sequential_10/conv2d_43/BiasAdd/ReadVariableOp?0sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp?-sequential_10/conv2d_43/Conv2D/ReadVariableOp?/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp?-sequential_10/dense_30/BiasAdd/ReadVariableOp?/sequential_10/dense_30/BiasAdd_1/ReadVariableOp?,sequential_10/dense_30/MatMul/ReadVariableOp?.sequential_10/dense_30/MatMul_1/ReadVariableOp?-sequential_10/dense_31/BiasAdd/ReadVariableOp?/sequential_10/dense_31/BiasAdd_1/ReadVariableOp?,sequential_10/dense_31/MatMul/ReadVariableOp?.sequential_10/dense_31/MatMul_1/ReadVariableOp?
-sequential_10/conv2d_40/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_10/conv2d_40/Conv2DConv2Dinputs_05sequential_10/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
.sequential_10/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_10/conv2d_40/BiasAddBiasAdd'sequential_10/conv2d_40/Conv2D:output:06sequential_10/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
sequential_10/conv2d_40/TanhTanh(sequential_10/conv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
&sequential_10/max_pooling2d_30/MaxPoolMaxPool sequential_10/conv2d_40/Tanh:y:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
k
&sequential_10/dropout_30/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
$sequential_10/dropout_30/dropout/MulMul/sequential_10/max_pooling2d_30/MaxPool:output:0/sequential_10/dropout_30/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
&sequential_10/dropout_30/dropout/ShapeShape/sequential_10/max_pooling2d_30/MaxPool:output:0*
T0*
_output_shapes
:?
=sequential_10/dropout_30/dropout/random_uniform/RandomUniformRandomUniform/sequential_10/dropout_30/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0t
/sequential_10/dropout_30/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
-sequential_10/dropout_30/dropout/GreaterEqualGreaterEqualFsequential_10/dropout_30/dropout/random_uniform/RandomUniform:output:08sequential_10/dropout_30/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
%sequential_10/dropout_30/dropout/CastCast1sequential_10/dropout_30/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
&sequential_10/dropout_30/dropout/Mul_1Mul(sequential_10/dropout_30/dropout/Mul:z:0)sequential_10/dropout_30/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
-sequential_10/conv2d_41/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
sequential_10/conv2d_41/Conv2DConv2D*sequential_10/dropout_30/dropout/Mul_1:z:05sequential_10/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
.sequential_10/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_10/conv2d_41/BiasAddBiasAdd'sequential_10/conv2d_41/Conv2D:output:06sequential_10/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
sequential_10/conv2d_41/TanhTanh(sequential_10/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @?
&sequential_10/max_pooling2d_31/MaxPoolMaxPool sequential_10/conv2d_41/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
k
&sequential_10/dropout_31/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
$sequential_10/dropout_31/dropout/MulMul/sequential_10/max_pooling2d_31/MaxPool:output:0/sequential_10/dropout_31/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@?
&sequential_10/dropout_31/dropout/ShapeShape/sequential_10/max_pooling2d_31/MaxPool:output:0*
T0*
_output_shapes
:?
=sequential_10/dropout_31/dropout/random_uniform/RandomUniformRandomUniform/sequential_10/dropout_31/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0t
/sequential_10/dropout_31/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
-sequential_10/dropout_31/dropout/GreaterEqualGreaterEqualFsequential_10/dropout_31/dropout/random_uniform/RandomUniform:output:08sequential_10/dropout_31/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
%sequential_10/dropout_31/dropout/CastCast1sequential_10/dropout_31/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
&sequential_10/dropout_31/dropout/Mul_1Mul(sequential_10/dropout_31/dropout/Mul:z:0)sequential_10/dropout_31/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@?
-sequential_10/conv2d_42/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
sequential_10/conv2d_42/Conv2DConv2D*sequential_10/dropout_31/dropout/Mul_1:z:05sequential_10/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
.sequential_10/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_10/conv2d_42/BiasAddBiasAdd'sequential_10/conv2d_42/Conv2D:output:06sequential_10/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
sequential_10/conv2d_42/TanhTanh(sequential_10/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
&sequential_10/max_pooling2d_32/MaxPoolMaxPool sequential_10/conv2d_42/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
k
&sequential_10/dropout_32/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
$sequential_10/dropout_32/dropout/MulMul/sequential_10/max_pooling2d_32/MaxPool:output:0/sequential_10/dropout_32/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@?
&sequential_10/dropout_32/dropout/ShapeShape/sequential_10/max_pooling2d_32/MaxPool:output:0*
T0*
_output_shapes
:?
=sequential_10/dropout_32/dropout/random_uniform/RandomUniformRandomUniform/sequential_10/dropout_32/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0t
/sequential_10/dropout_32/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
-sequential_10/dropout_32/dropout/GreaterEqualGreaterEqualFsequential_10/dropout_32/dropout/random_uniform/RandomUniform:output:08sequential_10/dropout_32/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
%sequential_10/dropout_32/dropout/CastCast1sequential_10/dropout_32/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
&sequential_10/dropout_32/dropout/Mul_1Mul(sequential_10/dropout_32/dropout/Mul:z:0)sequential_10/dropout_32/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@?
-sequential_10/conv2d_43/Conv2D/ReadVariableOpReadVariableOp6sequential_10_conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential_10/conv2d_43/Conv2DConv2D*sequential_10/dropout_32/dropout/Mul_1:z:05sequential_10/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
.sequential_10/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp7sequential_10_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_10/conv2d_43/BiasAddBiasAdd'sequential_10/conv2d_43/Conv2D:output:06sequential_10/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
sequential_10/conv2d_43/TanhTanh(sequential_10/conv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:??????????o
sequential_10/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
 sequential_10/flatten_10/ReshapeReshape sequential_10/conv2d_43/Tanh:y:0'sequential_10/flatten_10/Const:output:0*
T0*(
_output_shapes
:??????????@?
,sequential_10/dense_30/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0?
sequential_10/dense_30/MatMulMatMul)sequential_10/flatten_10/Reshape:output:04sequential_10/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
-sequential_10/dense_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
sequential_10/dense_30/BiasAddBiasAdd'sequential_10/dense_30/MatMul:product:05sequential_10/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 
sequential_10/dense_30/TanhTanh'sequential_10/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? ?
,sequential_10/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0?
sequential_10/dense_31/MatMulMatMulsequential_10/dense_30/Tanh:y:04sequential_10/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_10/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_10/dense_31/BiasAddBiasAdd'sequential_10/dense_31/MatMul:product:05sequential_10/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential_10/dense_31/SigmoidSigmoid'sequential_10/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
/sequential_10/conv2d_40/Conv2D_1/ReadVariableOpReadVariableOp6sequential_10_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
 sequential_10/conv2d_40/Conv2D_1Conv2Dinputs_17sequential_10/conv2d_40/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
0sequential_10/conv2d_40/BiasAdd_1/ReadVariableOpReadVariableOp7sequential_10_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
!sequential_10/conv2d_40/BiasAdd_1BiasAdd)sequential_10/conv2d_40/Conv2D_1:output:08sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
sequential_10/conv2d_40/Tanh_1Tanh*sequential_10/conv2d_40/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????@@?
(sequential_10/max_pooling2d_30/MaxPool_1MaxPool"sequential_10/conv2d_40/Tanh_1:y:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
m
(sequential_10/dropout_30/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
&sequential_10/dropout_30/dropout_1/MulMul1sequential_10/max_pooling2d_30/MaxPool_1:output:01sequential_10/dropout_30/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????  ?
(sequential_10/dropout_30/dropout_1/ShapeShape1sequential_10/max_pooling2d_30/MaxPool_1:output:0*
T0*
_output_shapes
:?
?sequential_10/dropout_30/dropout_1/random_uniform/RandomUniformRandomUniform1sequential_10/dropout_30/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0v
1sequential_10/dropout_30/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
/sequential_10/dropout_30/dropout_1/GreaterEqualGreaterEqualHsequential_10/dropout_30/dropout_1/random_uniform/RandomUniform:output:0:sequential_10/dropout_30/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
'sequential_10/dropout_30/dropout_1/CastCast3sequential_10/dropout_30/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
(sequential_10/dropout_30/dropout_1/Mul_1Mul*sequential_10/dropout_30/dropout_1/Mul:z:0+sequential_10/dropout_30/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
/sequential_10/conv2d_41/Conv2D_1/ReadVariableOpReadVariableOp6sequential_10_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
 sequential_10/conv2d_41/Conv2D_1Conv2D,sequential_10/dropout_30/dropout_1/Mul_1:z:07sequential_10/conv2d_41/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
0sequential_10/conv2d_41/BiasAdd_1/ReadVariableOpReadVariableOp7sequential_10_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
!sequential_10/conv2d_41/BiasAdd_1BiasAdd)sequential_10/conv2d_41/Conv2D_1:output:08sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
sequential_10/conv2d_41/Tanh_1Tanh*sequential_10/conv2d_41/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????  @?
(sequential_10/max_pooling2d_31/MaxPool_1MaxPool"sequential_10/conv2d_41/Tanh_1:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
m
(sequential_10/dropout_31/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
&sequential_10/dropout_31/dropout_1/MulMul1sequential_10/max_pooling2d_31/MaxPool_1:output:01sequential_10/dropout_31/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????@?
(sequential_10/dropout_31/dropout_1/ShapeShape1sequential_10/max_pooling2d_31/MaxPool_1:output:0*
T0*
_output_shapes
:?
?sequential_10/dropout_31/dropout_1/random_uniform/RandomUniformRandomUniform1sequential_10/dropout_31/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0v
1sequential_10/dropout_31/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
/sequential_10/dropout_31/dropout_1/GreaterEqualGreaterEqualHsequential_10/dropout_31/dropout_1/random_uniform/RandomUniform:output:0:sequential_10/dropout_31/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
'sequential_10/dropout_31/dropout_1/CastCast3sequential_10/dropout_31/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
(sequential_10/dropout_31/dropout_1/Mul_1Mul*sequential_10/dropout_31/dropout_1/Mul:z:0+sequential_10/dropout_31/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????@?
/sequential_10/conv2d_42/Conv2D_1/ReadVariableOpReadVariableOp6sequential_10_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
 sequential_10/conv2d_42/Conv2D_1Conv2D,sequential_10/dropout_31/dropout_1/Mul_1:z:07sequential_10/conv2d_42/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
0sequential_10/conv2d_42/BiasAdd_1/ReadVariableOpReadVariableOp7sequential_10_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
!sequential_10/conv2d_42/BiasAdd_1BiasAdd)sequential_10/conv2d_42/Conv2D_1:output:08sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
sequential_10/conv2d_42/Tanh_1Tanh*sequential_10/conv2d_42/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????@?
(sequential_10/max_pooling2d_32/MaxPool_1MaxPool"sequential_10/conv2d_42/Tanh_1:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
m
(sequential_10/dropout_32/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
&sequential_10/dropout_32/dropout_1/MulMul1sequential_10/max_pooling2d_32/MaxPool_1:output:01sequential_10/dropout_32/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????@?
(sequential_10/dropout_32/dropout_1/ShapeShape1sequential_10/max_pooling2d_32/MaxPool_1:output:0*
T0*
_output_shapes
:?
?sequential_10/dropout_32/dropout_1/random_uniform/RandomUniformRandomUniform1sequential_10/dropout_32/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0v
1sequential_10/dropout_32/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
/sequential_10/dropout_32/dropout_1/GreaterEqualGreaterEqualHsequential_10/dropout_32/dropout_1/random_uniform/RandomUniform:output:0:sequential_10/dropout_32/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
'sequential_10/dropout_32/dropout_1/CastCast3sequential_10/dropout_32/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
(sequential_10/dropout_32/dropout_1/Mul_1Mul*sequential_10/dropout_32/dropout_1/Mul:z:0+sequential_10/dropout_32/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????@?
/sequential_10/conv2d_43/Conv2D_1/ReadVariableOpReadVariableOp6sequential_10_conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
 sequential_10/conv2d_43/Conv2D_1Conv2D,sequential_10/dropout_32/dropout_1/Mul_1:z:07sequential_10/conv2d_43/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
0sequential_10/conv2d_43/BiasAdd_1/ReadVariableOpReadVariableOp7sequential_10_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!sequential_10/conv2d_43/BiasAdd_1BiasAdd)sequential_10/conv2d_43/Conv2D_1:output:08sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
sequential_10/conv2d_43/Tanh_1Tanh*sequential_10/conv2d_43/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????q
 sequential_10/flatten_10/Const_1Const*
_output_shapes
:*
dtype0*
valueB"????    ?
"sequential_10/flatten_10/Reshape_1Reshape"sequential_10/conv2d_43/Tanh_1:y:0)sequential_10/flatten_10/Const_1:output:0*
T0*(
_output_shapes
:??????????@?
.sequential_10/dense_30/MatMul_1/ReadVariableOpReadVariableOp5sequential_10_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0?
sequential_10/dense_30/MatMul_1MatMul+sequential_10/flatten_10/Reshape_1:output:06sequential_10/dense_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
/sequential_10/dense_30/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_10_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
 sequential_10/dense_30/BiasAdd_1BiasAdd)sequential_10/dense_30/MatMul_1:product:07sequential_10/dense_30/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
sequential_10/dense_30/Tanh_1Tanh)sequential_10/dense_30/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????? ?
.sequential_10/dense_31/MatMul_1/ReadVariableOpReadVariableOp5sequential_10_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0?
sequential_10/dense_31/MatMul_1MatMul!sequential_10/dense_30/Tanh_1:y:06sequential_10/dense_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
/sequential_10/dense_31/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_10_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
 sequential_10/dense_31/BiasAdd_1BiasAdd)sequential_10/dense_31/MatMul_1:product:07sequential_10/dense_31/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 sequential_10/dense_31/Sigmoid_1Sigmoid)sequential_10/dense_31/BiasAdd_1:output:0*
T0*(
_output_shapes
:???????????
distance_layer_10/subSub"sequential_10/dense_31/Sigmoid:y:0$sequential_10/dense_31/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????p
distance_layer_10/SquareSquaredistance_layer_10/sub:z:0*
T0*(
_output_shapes
:??????????i
'distance_layer_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
distance_layer_10/SumSumdistance_layer_10/Square:y:00distance_layer_10/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(`
distance_layer_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
distance_layer_10/MaximumMaximumdistance_layer_10/Sum:output:0$distance_layer_10/Maximum/y:output:0*
T0*'
_output_shapes
:?????????\
distance_layer_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
distance_layer_10/Maximum_1Maximumdistance_layer_10/Maximum:z:0 distance_layer_10/Const:output:0*
T0*'
_output_shapes
:?????????q
distance_layer_10/SqrtSqrtdistance_layer_10/Maximum_1:z:0*
T0*'
_output_shapes
:??????????
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_32/MatMulMatMuldistance_layer_10/Sqrt:y:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_32/SigmoidSigmoiddense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_32/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????

NoOpNoOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp/^sequential_10/conv2d_40/BiasAdd/ReadVariableOp1^sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp.^sequential_10/conv2d_40/Conv2D/ReadVariableOp0^sequential_10/conv2d_40/Conv2D_1/ReadVariableOp/^sequential_10/conv2d_41/BiasAdd/ReadVariableOp1^sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp.^sequential_10/conv2d_41/Conv2D/ReadVariableOp0^sequential_10/conv2d_41/Conv2D_1/ReadVariableOp/^sequential_10/conv2d_42/BiasAdd/ReadVariableOp1^sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp.^sequential_10/conv2d_42/Conv2D/ReadVariableOp0^sequential_10/conv2d_42/Conv2D_1/ReadVariableOp/^sequential_10/conv2d_43/BiasAdd/ReadVariableOp1^sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp.^sequential_10/conv2d_43/Conv2D/ReadVariableOp0^sequential_10/conv2d_43/Conv2D_1/ReadVariableOp.^sequential_10/dense_30/BiasAdd/ReadVariableOp0^sequential_10/dense_30/BiasAdd_1/ReadVariableOp-^sequential_10/dense_30/MatMul/ReadVariableOp/^sequential_10/dense_30/MatMul_1/ReadVariableOp.^sequential_10/dense_31/BiasAdd/ReadVariableOp0^sequential_10/dense_31/BiasAdd_1/ReadVariableOp-^sequential_10/dense_31/MatMul/ReadVariableOp/^sequential_10/dense_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2`
.sequential_10/conv2d_40/BiasAdd/ReadVariableOp.sequential_10/conv2d_40/BiasAdd/ReadVariableOp2d
0sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp0sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp2^
-sequential_10/conv2d_40/Conv2D/ReadVariableOp-sequential_10/conv2d_40/Conv2D/ReadVariableOp2b
/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp2`
.sequential_10/conv2d_41/BiasAdd/ReadVariableOp.sequential_10/conv2d_41/BiasAdd/ReadVariableOp2d
0sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp0sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp2^
-sequential_10/conv2d_41/Conv2D/ReadVariableOp-sequential_10/conv2d_41/Conv2D/ReadVariableOp2b
/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp2`
.sequential_10/conv2d_42/BiasAdd/ReadVariableOp.sequential_10/conv2d_42/BiasAdd/ReadVariableOp2d
0sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp0sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp2^
-sequential_10/conv2d_42/Conv2D/ReadVariableOp-sequential_10/conv2d_42/Conv2D/ReadVariableOp2b
/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp2`
.sequential_10/conv2d_43/BiasAdd/ReadVariableOp.sequential_10/conv2d_43/BiasAdd/ReadVariableOp2d
0sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp0sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp2^
-sequential_10/conv2d_43/Conv2D/ReadVariableOp-sequential_10/conv2d_43/Conv2D/ReadVariableOp2b
/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp2^
-sequential_10/dense_30/BiasAdd/ReadVariableOp-sequential_10/dense_30/BiasAdd/ReadVariableOp2b
/sequential_10/dense_30/BiasAdd_1/ReadVariableOp/sequential_10/dense_30/BiasAdd_1/ReadVariableOp2\
,sequential_10/dense_30/MatMul/ReadVariableOp,sequential_10/dense_30/MatMul/ReadVariableOp2`
.sequential_10/dense_30/MatMul_1/ReadVariableOp.sequential_10/dense_30/MatMul_1/ReadVariableOp2^
-sequential_10/dense_31/BiasAdd/ReadVariableOp-sequential_10/dense_31/BiasAdd/ReadVariableOp2b
/sequential_10/dense_31/BiasAdd_1/ReadVariableOp/sequential_10/dense_31/BiasAdd_1/ReadVariableOp2\
,sequential_10/dense_31/MatMul/ReadVariableOp,sequential_10/dense_31/MatMul/ReadVariableOp2`
.sequential_10/dense_31/MatMul_1/ReadVariableOp.sequential_10/dense_31/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
?

?
E__inference_conv2d_42_layer_call_and_return_conditional_losses_139123

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?:
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_139569
conv2d_40_input*
conv2d_40_139531:
conv2d_40_139533:*
conv2d_41_139538:@
conv2d_41_139540:@*
conv2d_42_139545:@@
conv2d_42_139547:@+
conv2d_43_139552:@?
conv2d_43_139554:	?#
dense_30_139558:
?@? 
dense_30_139560:	? #
dense_31_139563:
? ?
dense_31_139565:	?
identity??!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?"dropout_30/StatefulPartitionedCall?"dropout_31/StatefulPartitionedCall?"dropout_32/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputconv2d_40_139531conv2d_40_139533*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_139073?
 max_pooling2d_30/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_139028?
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_139346?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0conv2d_41_139538conv2d_41_139540*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_139098?
 max_pooling2d_31/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_139040?
"dropout_31/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_31/PartitionedCall:output:0#^dropout_30/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_31_layer_call_and_return_conditional_losses_139313?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall+dropout_31/StatefulPartitionedCall:output:0conv2d_42_139545conv2d_42_139547*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_139123?
 max_pooling2d_32/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_139052?
"dropout_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0#^dropout_31/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_139280?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_32/StatefulPartitionedCall:output:0conv2d_43_139552conv2d_43_139554*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_139148?
flatten_10/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_139160?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_30_139558dense_30_139560*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_139173?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_139563dense_31_139565*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_139190y
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall#^dropout_31/StatefulPartitionedCall#^dropout_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall2H
"dropout_31/StatefulPartitionedCall"dropout_31/StatefulPartitionedCall2H
"dropout_32/StatefulPartitionedCall"dropout_32/StatefulPartitionedCall:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_40_input
?
M
1__inference_max_pooling2d_31_layer_call_fn_140626

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_139040?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_139040

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_41_layer_call_fn_140610

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_139098w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
E__inference_conv2d_41_layer_call_and_return_conditional_losses_140621

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????  @_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_30_layer_call_fn_140569

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_139028?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

v
M__inference_distance_layer_10_layer_call_and_return_conditional_losses_139630

anchor
compare
identityN
subSubanchorcompare*
T0*(
_output_shapes
:??????????L
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????M
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????P
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameanchor:QM
(
_output_shapes
:??????????
!
_user_specified_name	compare
?
h
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_140631

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_10_layer_call_and_return_conditional_losses_139160

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

e
F__inference_dropout_32_layer_call_and_return_conditional_losses_139280

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
F__inference_dropout_30_layer_call_and_return_conditional_losses_140589

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?X
?	
I__inference_sequential_10_layer_call_and_return_conditional_losses_140504

inputsB
(conv2d_40_conv2d_readvariableop_resource:7
)conv2d_40_biasadd_readvariableop_resource:B
(conv2d_41_conv2d_readvariableop_resource:@7
)conv2d_41_biasadd_readvariableop_resource:@B
(conv2d_42_conv2d_readvariableop_resource:@@7
)conv2d_42_biasadd_readvariableop_resource:@C
(conv2d_43_conv2d_readvariableop_resource:@?8
)conv2d_43_biasadd_readvariableop_resource:	?;
'dense_30_matmul_readvariableop_resource:
?@? 7
(dense_30_biasadd_readvariableop_resource:	? ;
'dense_31_matmul_readvariableop_resource:
? ?7
(dense_31_biasadd_readvariableop_resource:	?
identity?? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@l
conv2d_40/TanhTanhconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
max_pooling2d_30/MaxPoolMaxPoolconv2d_40/Tanh:y:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
]
dropout_30/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_30/dropout/MulMul!max_pooling2d_30/MaxPool:output:0!dropout_30/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  i
dropout_30/dropout/ShapeShape!max_pooling2d_30/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_30/dropout/random_uniform/RandomUniformRandomUniform!dropout_30/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0f
!dropout_30/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_30/dropout/GreaterEqualGreaterEqual8dropout_30/dropout/random_uniform/RandomUniform:output:0*dropout_30/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
dropout_30/dropout/CastCast#dropout_30/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
dropout_30/dropout/Mul_1Muldropout_30/dropout/Mul:z:0dropout_30/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_41/Conv2DConv2Ddropout_30/dropout/Mul_1:z:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @l
conv2d_41/TanhTanhconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @?
max_pooling2d_31/MaxPoolMaxPoolconv2d_41/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
]
dropout_31/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_31/dropout/MulMul!max_pooling2d_31/MaxPool:output:0!dropout_31/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@i
dropout_31/dropout/ShapeShape!max_pooling2d_31/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_31/dropout/random_uniform/RandomUniformRandomUniform!dropout_31/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0f
!dropout_31/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_31/dropout/GreaterEqualGreaterEqual8dropout_31/dropout/random_uniform/RandomUniform:output:0*dropout_31/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
dropout_31/dropout/CastCast#dropout_31/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
dropout_31/dropout/Mul_1Muldropout_31/dropout/Mul:z:0dropout_31/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_42/Conv2DConv2Ddropout_31/dropout/Mul_1:z:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_42/TanhTanhconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_32/MaxPoolMaxPoolconv2d_42/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
]
dropout_32/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_32/dropout/MulMul!max_pooling2d_32/MaxPool:output:0!dropout_32/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@i
dropout_32/dropout/ShapeShape!max_pooling2d_32/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_32/dropout/random_uniform/RandomUniformRandomUniform!dropout_32/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0f
!dropout_32/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_32/dropout/GreaterEqualGreaterEqual8dropout_32/dropout/random_uniform/RandomUniform:output:0*dropout_32/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
dropout_32/dropout/CastCast#dropout_32/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
dropout_32/dropout/Mul_1Muldropout_32/dropout/Mul:z:0dropout_32/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_43/Conv2DConv2Ddropout_32/dropout/Mul_1:z:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_43/TanhTanhconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:??????????a
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    
flatten_10/ReshapeReshapeconv2d_43/Tanh:y:0flatten_10/Const:output:0*
T0*(
_output_shapes
:??????????@?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0?
dense_30/MatMulMatMulflatten_10/Reshape:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? c
dense_30/TanhTanhdense_30/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? ?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0?
dense_31/MatMulMatMuldense_30/Tanh:y:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????i
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*(
_output_shapes
:??????????d
IdentityIdentitydense_31/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
d
F__inference_dropout_32_layer_call_and_return_conditional_losses_139135

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
G
+__inference_dropout_31_layer_call_fn_140636

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_31_layer_call_and_return_conditional_losses_139110h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv2d_43_layer_call_and_return_conditional_losses_139148

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
TanhTanhBiasAdd:output:0*
T0*0
_output_shapes
:??????????`
IdentityIdentityTanh:y:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_dense_31_layer_call_fn_140775

inputs
unknown:
? ?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_139190p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_139019

anchor
compareY
?model_10_sequential_10_conv2d_40_conv2d_readvariableop_resource:N
@model_10_sequential_10_conv2d_40_biasadd_readvariableop_resource:Y
?model_10_sequential_10_conv2d_41_conv2d_readvariableop_resource:@N
@model_10_sequential_10_conv2d_41_biasadd_readvariableop_resource:@Y
?model_10_sequential_10_conv2d_42_conv2d_readvariableop_resource:@@N
@model_10_sequential_10_conv2d_42_biasadd_readvariableop_resource:@Z
?model_10_sequential_10_conv2d_43_conv2d_readvariableop_resource:@?O
@model_10_sequential_10_conv2d_43_biasadd_readvariableop_resource:	?R
>model_10_sequential_10_dense_30_matmul_readvariableop_resource:
?@? N
?model_10_sequential_10_dense_30_biasadd_readvariableop_resource:	? R
>model_10_sequential_10_dense_31_matmul_readvariableop_resource:
? ?N
?model_10_sequential_10_dense_31_biasadd_readvariableop_resource:	?B
0model_10_dense_32_matmul_readvariableop_resource:?
1model_10_dense_32_biasadd_readvariableop_resource:
identity??(model_10/dense_32/BiasAdd/ReadVariableOp?'model_10/dense_32/MatMul/ReadVariableOp?7model_10/sequential_10/conv2d_40/BiasAdd/ReadVariableOp?9model_10/sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp?6model_10/sequential_10/conv2d_40/Conv2D/ReadVariableOp?8model_10/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp?7model_10/sequential_10/conv2d_41/BiasAdd/ReadVariableOp?9model_10/sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp?6model_10/sequential_10/conv2d_41/Conv2D/ReadVariableOp?8model_10/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp?7model_10/sequential_10/conv2d_42/BiasAdd/ReadVariableOp?9model_10/sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp?6model_10/sequential_10/conv2d_42/Conv2D/ReadVariableOp?8model_10/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp?7model_10/sequential_10/conv2d_43/BiasAdd/ReadVariableOp?9model_10/sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp?6model_10/sequential_10/conv2d_43/Conv2D/ReadVariableOp?8model_10/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp?6model_10/sequential_10/dense_30/BiasAdd/ReadVariableOp?8model_10/sequential_10/dense_30/BiasAdd_1/ReadVariableOp?5model_10/sequential_10/dense_30/MatMul/ReadVariableOp?7model_10/sequential_10/dense_30/MatMul_1/ReadVariableOp?6model_10/sequential_10/dense_31/BiasAdd/ReadVariableOp?8model_10/sequential_10/dense_31/BiasAdd_1/ReadVariableOp?5model_10/sequential_10/dense_31/MatMul/ReadVariableOp?7model_10/sequential_10/dense_31/MatMul_1/ReadVariableOp?
6model_10/sequential_10/conv2d_40/Conv2D/ReadVariableOpReadVariableOp?model_10_sequential_10_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
'model_10/sequential_10/conv2d_40/Conv2DConv2Danchor>model_10/sequential_10/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
7model_10/sequential_10/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp@model_10_sequential_10_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(model_10/sequential_10/conv2d_40/BiasAddBiasAdd0model_10/sequential_10/conv2d_40/Conv2D:output:0?model_10/sequential_10/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
%model_10/sequential_10/conv2d_40/TanhTanh1model_10/sequential_10/conv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
/model_10/sequential_10/max_pooling2d_30/MaxPoolMaxPool)model_10/sequential_10/conv2d_40/Tanh:y:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
?
*model_10/sequential_10/dropout_30/IdentityIdentity8model_10/sequential_10/max_pooling2d_30/MaxPool:output:0*
T0*/
_output_shapes
:?????????  ?
6model_10/sequential_10/conv2d_41/Conv2D/ReadVariableOpReadVariableOp?model_10_sequential_10_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
'model_10/sequential_10/conv2d_41/Conv2DConv2D3model_10/sequential_10/dropout_30/Identity:output:0>model_10/sequential_10/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
7model_10/sequential_10/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp@model_10_sequential_10_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
(model_10/sequential_10/conv2d_41/BiasAddBiasAdd0model_10/sequential_10/conv2d_41/Conv2D:output:0?model_10/sequential_10/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
%model_10/sequential_10/conv2d_41/TanhTanh1model_10/sequential_10/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @?
/model_10/sequential_10/max_pooling2d_31/MaxPoolMaxPool)model_10/sequential_10/conv2d_41/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
*model_10/sequential_10/dropout_31/IdentityIdentity8model_10/sequential_10/max_pooling2d_31/MaxPool:output:0*
T0*/
_output_shapes
:?????????@?
6model_10/sequential_10/conv2d_42/Conv2D/ReadVariableOpReadVariableOp?model_10_sequential_10_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
'model_10/sequential_10/conv2d_42/Conv2DConv2D3model_10/sequential_10/dropout_31/Identity:output:0>model_10/sequential_10/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
7model_10/sequential_10/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp@model_10_sequential_10_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
(model_10/sequential_10/conv2d_42/BiasAddBiasAdd0model_10/sequential_10/conv2d_42/Conv2D:output:0?model_10/sequential_10/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
%model_10/sequential_10/conv2d_42/TanhTanh1model_10/sequential_10/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
/model_10/sequential_10/max_pooling2d_32/MaxPoolMaxPool)model_10/sequential_10/conv2d_42/Tanh:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
*model_10/sequential_10/dropout_32/IdentityIdentity8model_10/sequential_10/max_pooling2d_32/MaxPool:output:0*
T0*/
_output_shapes
:?????????@?
6model_10/sequential_10/conv2d_43/Conv2D/ReadVariableOpReadVariableOp?model_10_sequential_10_conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
'model_10/sequential_10/conv2d_43/Conv2DConv2D3model_10/sequential_10/dropout_32/Identity:output:0>model_10/sequential_10/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
7model_10/sequential_10/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp@model_10_sequential_10_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
(model_10/sequential_10/conv2d_43/BiasAddBiasAdd0model_10/sequential_10/conv2d_43/Conv2D:output:0?model_10/sequential_10/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
%model_10/sequential_10/conv2d_43/TanhTanh1model_10/sequential_10/conv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:??????????x
'model_10/sequential_10/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)model_10/sequential_10/flatten_10/ReshapeReshape)model_10/sequential_10/conv2d_43/Tanh:y:00model_10/sequential_10/flatten_10/Const:output:0*
T0*(
_output_shapes
:??????????@?
5model_10/sequential_10/dense_30/MatMul/ReadVariableOpReadVariableOp>model_10_sequential_10_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0?
&model_10/sequential_10/dense_30/MatMulMatMul2model_10/sequential_10/flatten_10/Reshape:output:0=model_10/sequential_10/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
6model_10/sequential_10/dense_30/BiasAdd/ReadVariableOpReadVariableOp?model_10_sequential_10_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
'model_10/sequential_10/dense_30/BiasAddBiasAdd0model_10/sequential_10/dense_30/MatMul:product:0>model_10/sequential_10/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
$model_10/sequential_10/dense_30/TanhTanh0model_10/sequential_10/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? ?
5model_10/sequential_10/dense_31/MatMul/ReadVariableOpReadVariableOp>model_10_sequential_10_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0?
&model_10/sequential_10/dense_31/MatMulMatMul(model_10/sequential_10/dense_30/Tanh:y:0=model_10/sequential_10/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
6model_10/sequential_10/dense_31/BiasAdd/ReadVariableOpReadVariableOp?model_10_sequential_10_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'model_10/sequential_10/dense_31/BiasAddBiasAdd0model_10/sequential_10/dense_31/MatMul:product:0>model_10/sequential_10/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'model_10/sequential_10/dense_31/SigmoidSigmoid0model_10/sequential_10/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
8model_10/sequential_10/conv2d_40/Conv2D_1/ReadVariableOpReadVariableOp?model_10_sequential_10_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
)model_10/sequential_10/conv2d_40/Conv2D_1Conv2Dcompare@model_10/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
9model_10/sequential_10/conv2d_40/BiasAdd_1/ReadVariableOpReadVariableOp@model_10_sequential_10_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*model_10/sequential_10/conv2d_40/BiasAdd_1BiasAdd2model_10/sequential_10/conv2d_40/Conv2D_1:output:0Amodel_10/sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
'model_10/sequential_10/conv2d_40/Tanh_1Tanh3model_10/sequential_10/conv2d_40/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????@@?
1model_10/sequential_10/max_pooling2d_30/MaxPool_1MaxPool+model_10/sequential_10/conv2d_40/Tanh_1:y:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
?
,model_10/sequential_10/dropout_30/Identity_1Identity:model_10/sequential_10/max_pooling2d_30/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????  ?
8model_10/sequential_10/conv2d_41/Conv2D_1/ReadVariableOpReadVariableOp?model_10_sequential_10_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
)model_10/sequential_10/conv2d_41/Conv2D_1Conv2D5model_10/sequential_10/dropout_30/Identity_1:output:0@model_10/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
9model_10/sequential_10/conv2d_41/BiasAdd_1/ReadVariableOpReadVariableOp@model_10_sequential_10_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
*model_10/sequential_10/conv2d_41/BiasAdd_1BiasAdd2model_10/sequential_10/conv2d_41/Conv2D_1:output:0Amodel_10/sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
'model_10/sequential_10/conv2d_41/Tanh_1Tanh3model_10/sequential_10/conv2d_41/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????  @?
1model_10/sequential_10/max_pooling2d_31/MaxPool_1MaxPool+model_10/sequential_10/conv2d_41/Tanh_1:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
,model_10/sequential_10/dropout_31/Identity_1Identity:model_10/sequential_10/max_pooling2d_31/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????@?
8model_10/sequential_10/conv2d_42/Conv2D_1/ReadVariableOpReadVariableOp?model_10_sequential_10_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
)model_10/sequential_10/conv2d_42/Conv2D_1Conv2D5model_10/sequential_10/dropout_31/Identity_1:output:0@model_10/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
9model_10/sequential_10/conv2d_42/BiasAdd_1/ReadVariableOpReadVariableOp@model_10_sequential_10_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
*model_10/sequential_10/conv2d_42/BiasAdd_1BiasAdd2model_10/sequential_10/conv2d_42/Conv2D_1:output:0Amodel_10/sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
'model_10/sequential_10/conv2d_42/Tanh_1Tanh3model_10/sequential_10/conv2d_42/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????@?
1model_10/sequential_10/max_pooling2d_32/MaxPool_1MaxPool+model_10/sequential_10/conv2d_42/Tanh_1:y:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
,model_10/sequential_10/dropout_32/Identity_1Identity:model_10/sequential_10/max_pooling2d_32/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????@?
8model_10/sequential_10/conv2d_43/Conv2D_1/ReadVariableOpReadVariableOp?model_10_sequential_10_conv2d_43_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
)model_10/sequential_10/conv2d_43/Conv2D_1Conv2D5model_10/sequential_10/dropout_32/Identity_1:output:0@model_10/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
9model_10/sequential_10/conv2d_43/BiasAdd_1/ReadVariableOpReadVariableOp@model_10_sequential_10_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*model_10/sequential_10/conv2d_43/BiasAdd_1BiasAdd2model_10/sequential_10/conv2d_43/Conv2D_1:output:0Amodel_10/sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
'model_10/sequential_10/conv2d_43/Tanh_1Tanh3model_10/sequential_10/conv2d_43/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????z
)model_10/sequential_10/flatten_10/Const_1Const*
_output_shapes
:*
dtype0*
valueB"????    ?
+model_10/sequential_10/flatten_10/Reshape_1Reshape+model_10/sequential_10/conv2d_43/Tanh_1:y:02model_10/sequential_10/flatten_10/Const_1:output:0*
T0*(
_output_shapes
:??????????@?
7model_10/sequential_10/dense_30/MatMul_1/ReadVariableOpReadVariableOp>model_10_sequential_10_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
?@? *
dtype0?
(model_10/sequential_10/dense_30/MatMul_1MatMul4model_10/sequential_10/flatten_10/Reshape_1:output:0?model_10/sequential_10/dense_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
8model_10/sequential_10/dense_30/BiasAdd_1/ReadVariableOpReadVariableOp?model_10_sequential_10_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
)model_10/sequential_10/dense_30/BiasAdd_1BiasAdd2model_10/sequential_10/dense_30/MatMul_1:product:0@model_10/sequential_10/dense_30/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
&model_10/sequential_10/dense_30/Tanh_1Tanh2model_10/sequential_10/dense_30/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????? ?
7model_10/sequential_10/dense_31/MatMul_1/ReadVariableOpReadVariableOp>model_10_sequential_10_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0?
(model_10/sequential_10/dense_31/MatMul_1MatMul*model_10/sequential_10/dense_30/Tanh_1:y:0?model_10/sequential_10/dense_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
8model_10/sequential_10/dense_31/BiasAdd_1/ReadVariableOpReadVariableOp?model_10_sequential_10_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)model_10/sequential_10/dense_31/BiasAdd_1BiasAdd2model_10/sequential_10/dense_31/MatMul_1:product:0@model_10/sequential_10/dense_31/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)model_10/sequential_10/dense_31/Sigmoid_1Sigmoid2model_10/sequential_10/dense_31/BiasAdd_1:output:0*
T0*(
_output_shapes
:???????????
model_10/distance_layer_10/subSub+model_10/sequential_10/dense_31/Sigmoid:y:0-model_10/sequential_10/dense_31/Sigmoid_1:y:0*
T0*(
_output_shapes
:???????????
!model_10/distance_layer_10/SquareSquare"model_10/distance_layer_10/sub:z:0*
T0*(
_output_shapes
:??????????r
0model_10/distance_layer_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
model_10/distance_layer_10/SumSum%model_10/distance_layer_10/Square:y:09model_10/distance_layer_10/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(i
$model_10/distance_layer_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
"model_10/distance_layer_10/MaximumMaximum'model_10/distance_layer_10/Sum:output:0-model_10/distance_layer_10/Maximum/y:output:0*
T0*'
_output_shapes
:?????????e
 model_10/distance_layer_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
$model_10/distance_layer_10/Maximum_1Maximum&model_10/distance_layer_10/Maximum:z:0)model_10/distance_layer_10/Const:output:0*
T0*'
_output_shapes
:??????????
model_10/distance_layer_10/SqrtSqrt(model_10/distance_layer_10/Maximum_1:z:0*
T0*'
_output_shapes
:??????????
'model_10/dense_32/MatMul/ReadVariableOpReadVariableOp0model_10_dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_10/dense_32/MatMulMatMul#model_10/distance_layer_10/Sqrt:y:0/model_10/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(model_10/dense_32/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_10/dense_32/BiasAddBiasAdd"model_10/dense_32/MatMul:product:00model_10/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
model_10/dense_32/SigmoidSigmoid"model_10/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
IdentityIdentitymodel_10/dense_32/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^model_10/dense_32/BiasAdd/ReadVariableOp(^model_10/dense_32/MatMul/ReadVariableOp8^model_10/sequential_10/conv2d_40/BiasAdd/ReadVariableOp:^model_10/sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp7^model_10/sequential_10/conv2d_40/Conv2D/ReadVariableOp9^model_10/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp8^model_10/sequential_10/conv2d_41/BiasAdd/ReadVariableOp:^model_10/sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp7^model_10/sequential_10/conv2d_41/Conv2D/ReadVariableOp9^model_10/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp8^model_10/sequential_10/conv2d_42/BiasAdd/ReadVariableOp:^model_10/sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp7^model_10/sequential_10/conv2d_42/Conv2D/ReadVariableOp9^model_10/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp8^model_10/sequential_10/conv2d_43/BiasAdd/ReadVariableOp:^model_10/sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp7^model_10/sequential_10/conv2d_43/Conv2D/ReadVariableOp9^model_10/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp7^model_10/sequential_10/dense_30/BiasAdd/ReadVariableOp9^model_10/sequential_10/dense_30/BiasAdd_1/ReadVariableOp6^model_10/sequential_10/dense_30/MatMul/ReadVariableOp8^model_10/sequential_10/dense_30/MatMul_1/ReadVariableOp7^model_10/sequential_10/dense_31/BiasAdd/ReadVariableOp9^model_10/sequential_10/dense_31/BiasAdd_1/ReadVariableOp6^model_10/sequential_10/dense_31/MatMul/ReadVariableOp8^model_10/sequential_10/dense_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 2T
(model_10/dense_32/BiasAdd/ReadVariableOp(model_10/dense_32/BiasAdd/ReadVariableOp2R
'model_10/dense_32/MatMul/ReadVariableOp'model_10/dense_32/MatMul/ReadVariableOp2r
7model_10/sequential_10/conv2d_40/BiasAdd/ReadVariableOp7model_10/sequential_10/conv2d_40/BiasAdd/ReadVariableOp2v
9model_10/sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp9model_10/sequential_10/conv2d_40/BiasAdd_1/ReadVariableOp2p
6model_10/sequential_10/conv2d_40/Conv2D/ReadVariableOp6model_10/sequential_10/conv2d_40/Conv2D/ReadVariableOp2t
8model_10/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp8model_10/sequential_10/conv2d_40/Conv2D_1/ReadVariableOp2r
7model_10/sequential_10/conv2d_41/BiasAdd/ReadVariableOp7model_10/sequential_10/conv2d_41/BiasAdd/ReadVariableOp2v
9model_10/sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp9model_10/sequential_10/conv2d_41/BiasAdd_1/ReadVariableOp2p
6model_10/sequential_10/conv2d_41/Conv2D/ReadVariableOp6model_10/sequential_10/conv2d_41/Conv2D/ReadVariableOp2t
8model_10/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp8model_10/sequential_10/conv2d_41/Conv2D_1/ReadVariableOp2r
7model_10/sequential_10/conv2d_42/BiasAdd/ReadVariableOp7model_10/sequential_10/conv2d_42/BiasAdd/ReadVariableOp2v
9model_10/sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp9model_10/sequential_10/conv2d_42/BiasAdd_1/ReadVariableOp2p
6model_10/sequential_10/conv2d_42/Conv2D/ReadVariableOp6model_10/sequential_10/conv2d_42/Conv2D/ReadVariableOp2t
8model_10/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp8model_10/sequential_10/conv2d_42/Conv2D_1/ReadVariableOp2r
7model_10/sequential_10/conv2d_43/BiasAdd/ReadVariableOp7model_10/sequential_10/conv2d_43/BiasAdd/ReadVariableOp2v
9model_10/sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp9model_10/sequential_10/conv2d_43/BiasAdd_1/ReadVariableOp2p
6model_10/sequential_10/conv2d_43/Conv2D/ReadVariableOp6model_10/sequential_10/conv2d_43/Conv2D/ReadVariableOp2t
8model_10/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp8model_10/sequential_10/conv2d_43/Conv2D_1/ReadVariableOp2p
6model_10/sequential_10/dense_30/BiasAdd/ReadVariableOp6model_10/sequential_10/dense_30/BiasAdd/ReadVariableOp2t
8model_10/sequential_10/dense_30/BiasAdd_1/ReadVariableOp8model_10/sequential_10/dense_30/BiasAdd_1/ReadVariableOp2n
5model_10/sequential_10/dense_30/MatMul/ReadVariableOp5model_10/sequential_10/dense_30/MatMul/ReadVariableOp2r
7model_10/sequential_10/dense_30/MatMul_1/ReadVariableOp7model_10/sequential_10/dense_30/MatMul_1/ReadVariableOp2p
6model_10/sequential_10/dense_31/BiasAdd/ReadVariableOp6model_10/sequential_10/dense_31/BiasAdd/ReadVariableOp2t
8model_10/sequential_10/dense_31/BiasAdd_1/ReadVariableOp8model_10/sequential_10/dense_31/BiasAdd_1/ReadVariableOp2n
5model_10/sequential_10/dense_31/MatMul/ReadVariableOp5model_10/sequential_10/dense_31/MatMul/ReadVariableOp2r
7model_10/sequential_10/dense_31/MatMul_1/ReadVariableOp7model_10/sequential_10/dense_31/MatMul_1/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameanchor:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	compare
?
d
F__inference_dropout_31_layer_call_and_return_conditional_losses_139110

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_42_layer_call_and_return_conditional_losses_140678

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@_
IdentityIdentityTanh:y:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?5
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_139528
conv2d_40_input*
conv2d_40_139490:
conv2d_40_139492:*
conv2d_41_139497:@
conv2d_41_139499:@*
conv2d_42_139504:@@
conv2d_42_139506:@+
conv2d_43_139511:@?
conv2d_43_139513:	?#
dense_30_139517:
?@? 
dense_30_139519:	? #
dense_31_139522:
? ?
dense_31_139524:	?
identity??!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_42/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputconv2d_40_139490conv2d_40_139492*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_40_layer_call_and_return_conditional_losses_139073?
 max_pooling2d_30/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_139028?
dropout_30/PartitionedCallPartitionedCall)max_pooling2d_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_139085?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_41_139497conv2d_41_139499*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_41_layer_call_and_return_conditional_losses_139098?
 max_pooling2d_31/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_139040?
dropout_31/PartitionedCallPartitionedCall)max_pooling2d_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_31_layer_call_and_return_conditional_losses_139110?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0conv2d_42_139504conv2d_42_139506*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_139123?
 max_pooling2d_32/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_139052?
dropout_32/PartitionedCallPartitionedCall)max_pooling2d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_32_layer_call_and_return_conditional_losses_139135?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_32/PartitionedCall:output:0conv2d_43_139511conv2d_43_139513*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_139148?
flatten_10/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_10_layer_call_and_return_conditional_losses_139160?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_30_139517dense_30_139519*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_139173?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_139522dense_31_139524*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_139190y
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_40_input
?
[
2__inference_distance_layer_10_layer_call_fn_140510

anchor
compare
identity?
PartitionedCallPartitionedCallanchorcompare*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_distance_layer_10_layer_call_and_return_conditional_losses_139630`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameanchor:QM
(
_output_shapes
:??????????
!
_user_specified_name	compare
?

?
D__inference_dense_32_layer_call_and_return_conditional_losses_140544

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_140574

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_140317

anchor
compare!
unknown:
	unknown_0:#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:
?@? 
	unknown_8:	? 
	unknown_9:
? ?

unknown_10:	?

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallanchorcompareunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_139019o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameanchor:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	compare
?
d
+__inference_dropout_30_layer_call_fn_140584

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_30_layer_call_and_return_conditional_losses_139346w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
.__inference_sequential_10_layer_call_fn_139487
conv2d_40_input!
unknown:
	unknown_0:#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:
?@? 
	unknown_8:	? 
	unknown_9:
? ?

unknown_10:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_139431p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_40_input
?

e
F__inference_dropout_30_layer_call_and_return_conditional_losses_140601

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????  C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

e
F__inference_dropout_32_layer_call_and_return_conditional_losses_140715

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_140688

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_model_10_layer_call_fn_139681

anchor
compare!
unknown:
	unknown_0:#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:
?@? 
	unknown_8:	? 
	unknown_9:
? ?

unknown_10:	?

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallanchorcompareunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_10_layer_call_and_return_conditional_losses_139650o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:?????????@@:?????????@@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameanchor:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	compare
?

?
D__inference_dense_31_layer_call_and_return_conditional_losses_140786

inputs2
matmul_readvariableop_resource:
? ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_43_layer_call_fn_140724

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_139148x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
anchor7
serving_default_anchor:0?????????@@
C
compare8
serving_default_compare:0?????????@@<
dense_320
StatefulPartitionedCall:0?????????tensorflow/serving/predict:֝
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer_with_weights-3
layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0iter

1beta_1

2beta_2
	3decay
4learning_rate(m?)m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?(v?)v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?"
	optimizer
?
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
(12
)13"
trackable_list_wrapper
?
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
(12
)13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_model_10_layer_call_fn_139681
)__inference_model_10_layer_call_fn_139987
)__inference_model_10_layer_call_fn_140021
)__inference_model_10_layer_call_fn_139849?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_model_10_layer_call_and_return_conditional_losses_140130
D__inference_model_10_layer_call_and_return_conditional_losses_140281
D__inference_model_10_layer_call_and_return_conditional_losses_139898
D__inference_model_10_layer_call_and_return_conditional_losses_139947?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_139019anchorcompare"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Fserving_default"
signature_map
?

5kernel
6bias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W_random_generator
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
?

7kernel
8bias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
?
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j_random_generator
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
?

9kernel
:bias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
?
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}_random_generator
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

;kernel
<bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

=kernel
>bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

?kernel
@bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
v
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11"
trackable_list_wrapper
v
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_sequential_10_layer_call_fn_139224
.__inference_sequential_10_layer_call_fn_140346
.__inference_sequential_10_layer_call_fn_140375
.__inference_sequential_10_layer_call_fn_139487?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_10_layer_call_and_return_conditional_losses_140429
I__inference_sequential_10_layer_call_and_return_conditional_losses_140504
I__inference_sequential_10_layer_call_and_return_conditional_losses_139528
I__inference_sequential_10_layer_call_and_return_conditional_losses_139569?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_distance_layer_10_layer_call_fn_140510?
???
FullArgSpec(
args ?
jself
janchor
	jcompare
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_distance_layer_10_layer_call_and_return_conditional_losses_140524?
???
FullArgSpec(
args ?
jself
janchor
	jcompare
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
!:2dense_32/kernel
:2dense_32/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_32_layer_call_fn_140533?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_32_layer_call_and_return_conditional_losses_140544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(2conv2d_40/kernel
:2conv2d_40/bias
*:(@2conv2d_41/kernel
:@2conv2d_41/bias
*:(@@2conv2d_42/kernel
:@2conv2d_42/bias
+:)@?2conv2d_43/kernel
:?2conv2d_43/bias
#:!
?@? 2dense_30/kernel
:? 2dense_30/bias
#:!
? ?2dense_31/kernel
:?2dense_31/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_140317anchorcompare"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_40_layer_call_fn_140553?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_40_layer_call_and_return_conditional_losses_140564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
?2?
1__inference_max_pooling2d_30_layer_call_fn_140569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_140574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
+__inference_dropout_30_layer_call_fn_140579
+__inference_dropout_30_layer_call_fn_140584?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_30_layer_call_and_return_conditional_losses_140589
F__inference_dropout_30_layer_call_and_return_conditional_losses_140601?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_41_layer_call_fn_140610?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_41_layer_call_and_return_conditional_losses_140621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
?2?
1__inference_max_pooling2d_31_layer_call_fn_140626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_140631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
+__inference_dropout_31_layer_call_fn_140636
+__inference_dropout_31_layer_call_fn_140641?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_31_layer_call_and_return_conditional_losses_140646
F__inference_dropout_31_layer_call_and_return_conditional_losses_140658?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_42_layer_call_fn_140667?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_42_layer_call_and_return_conditional_losses_140678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
?2?
1__inference_max_pooling2d_32_layer_call_fn_140683?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_140688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
+__inference_dropout_32_layer_call_fn_140693
+__inference_dropout_32_layer_call_fn_140698?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_32_layer_call_and_return_conditional_losses_140703
F__inference_dropout_32_layer_call_and_return_conditional_losses_140715?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_43_layer_call_fn_140724?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_43_layer_call_and_return_conditional_losses_140735?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_flatten_10_layer_call_fn_140740?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_10_layer_call_and_return_conditional_losses_140746?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_30_layer_call_fn_140755?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_30_layer_call_and_return_conditional_losses_140766?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_31_layer_call_fn_140775?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_31_layer_call_and_return_conditional_losses_140786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
&:$2Adam/dense_32/kernel/m
 :2Adam/dense_32/bias/m
/:-2Adam/conv2d_40/kernel/m
!:2Adam/conv2d_40/bias/m
/:-@2Adam/conv2d_41/kernel/m
!:@2Adam/conv2d_41/bias/m
/:-@@2Adam/conv2d_42/kernel/m
!:@2Adam/conv2d_42/bias/m
0:.@?2Adam/conv2d_43/kernel/m
": ?2Adam/conv2d_43/bias/m
(:&
?@? 2Adam/dense_30/kernel/m
!:? 2Adam/dense_30/bias/m
(:&
? ?2Adam/dense_31/kernel/m
!:?2Adam/dense_31/bias/m
&:$2Adam/dense_32/kernel/v
 :2Adam/dense_32/bias/v
/:-2Adam/conv2d_40/kernel/v
!:2Adam/conv2d_40/bias/v
/:-@2Adam/conv2d_41/kernel/v
!:@2Adam/conv2d_41/bias/v
/:-@@2Adam/conv2d_42/kernel/v
!:@2Adam/conv2d_42/bias/v
0:.@?2Adam/conv2d_43/kernel/v
": ?2Adam/conv2d_43/bias/v
(:&
?@? 2Adam/dense_30/kernel/v
!:? 2Adam/dense_30/bias/v
(:&
? ?2Adam/dense_31/kernel/v
!:?2Adam/dense_31/bias/v?
!__inference__wrapped_model_139019?56789:;<=>?@()g?d
]?Z
X?U
(?%
anchor?????????@@
)?&
compare?????????@@
? "3?0
.
dense_32"?
dense_32??????????
E__inference_conv2d_40_layer_call_and_return_conditional_losses_140564l567?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
*__inference_conv2d_40_layer_call_fn_140553_567?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
E__inference_conv2d_41_layer_call_and_return_conditional_losses_140621l787?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  @
? ?
*__inference_conv2d_41_layer_call_fn_140610_787?4
-?*
(?%
inputs?????????  
? " ??????????  @?
E__inference_conv2d_42_layer_call_and_return_conditional_losses_140678l9:7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_42_layer_call_fn_140667_9:7?4
-?*
(?%
inputs?????????@
? " ??????????@?
E__inference_conv2d_43_layer_call_and_return_conditional_losses_140735m;<7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_43_layer_call_fn_140724`;<7?4
-?*
(?%
inputs?????????@
? "!????????????
D__inference_dense_30_layer_call_and_return_conditional_losses_140766^=>0?-
&?#
!?
inputs??????????@
? "&?#
?
0?????????? 
? ~
)__inference_dense_30_layer_call_fn_140755Q=>0?-
&?#
!?
inputs??????????@
? "??????????? ?
D__inference_dense_31_layer_call_and_return_conditional_losses_140786^?@0?-
&?#
!?
inputs?????????? 
? "&?#
?
0??????????
? ~
)__inference_dense_31_layer_call_fn_140775Q?@0?-
&?#
!?
inputs?????????? 
? "????????????
D__inference_dense_32_layer_call_and_return_conditional_losses_140544\()/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_32_layer_call_fn_140533O()/?,
%?"
 ?
inputs?????????
? "???????????
M__inference_distance_layer_10_layer_call_and_return_conditional_losses_140524}T?Q
J?G
!?
anchor??????????
"?
compare??????????
? "%?"
?
0?????????
? ?
2__inference_distance_layer_10_layer_call_fn_140510pT?Q
J?G
!?
anchor??????????
"?
compare??????????
? "???????????
F__inference_dropout_30_layer_call_and_return_conditional_losses_140589l;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
F__inference_dropout_30_layer_call_and_return_conditional_losses_140601l;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
+__inference_dropout_30_layer_call_fn_140579_;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
+__inference_dropout_30_layer_call_fn_140584_;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
F__inference_dropout_31_layer_call_and_return_conditional_losses_140646l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
F__inference_dropout_31_layer_call_and_return_conditional_losses_140658l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
+__inference_dropout_31_layer_call_fn_140636_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
+__inference_dropout_31_layer_call_fn_140641_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
F__inference_dropout_32_layer_call_and_return_conditional_losses_140703l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
F__inference_dropout_32_layer_call_and_return_conditional_losses_140715l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
+__inference_dropout_32_layer_call_fn_140693_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
+__inference_dropout_32_layer_call_fn_140698_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
F__inference_flatten_10_layer_call_and_return_conditional_losses_140746b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????@
? ?
+__inference_flatten_10_layer_call_fn_140740U8?5
.?+
)?&
inputs??????????
? "???????????@?
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_140574?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_30_layer_call_fn_140569?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_140631?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_31_layer_call_fn_140626?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_140688?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_32_layer_call_fn_140683?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_model_10_layer_call_and_return_conditional_losses_139898?56789:;<=>?@()o?l
e?b
X?U
(?%
anchor?????????@@
)?&
compare?????????@@
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_10_layer_call_and_return_conditional_losses_139947?56789:;<=>?@()o?l
e?b
X?U
(?%
anchor?????????@@
)?&
compare?????????@@
p

 
? "%?"
?
0?????????
? ?
D__inference_model_10_layer_call_and_return_conditional_losses_140130?56789:;<=>?@()r?o
h?e
[?X
*?'
inputs/0?????????@@
*?'
inputs/1?????????@@
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_10_layer_call_and_return_conditional_losses_140281?56789:;<=>?@()r?o
h?e
[?X
*?'
inputs/0?????????@@
*?'
inputs/1?????????@@
p

 
? "%?"
?
0?????????
? ?
)__inference_model_10_layer_call_fn_139681?56789:;<=>?@()o?l
e?b
X?U
(?%
anchor?????????@@
)?&
compare?????????@@
p 

 
? "???????????
)__inference_model_10_layer_call_fn_139849?56789:;<=>?@()o?l
e?b
X?U
(?%
anchor?????????@@
)?&
compare?????????@@
p

 
? "???????????
)__inference_model_10_layer_call_fn_139987?56789:;<=>?@()r?o
h?e
[?X
*?'
inputs/0?????????@@
*?'
inputs/1?????????@@
p 

 
? "???????????
)__inference_model_10_layer_call_fn_140021?56789:;<=>?@()r?o
h?e
[?X
*?'
inputs/0?????????@@
*?'
inputs/1?????????@@
p

 
? "???????????
I__inference_sequential_10_layer_call_and_return_conditional_losses_139528?56789:;<=>?@H?E
>?;
1?.
conv2d_40_input?????????@@
p 

 
? "&?#
?
0??????????
? ?
I__inference_sequential_10_layer_call_and_return_conditional_losses_139569?56789:;<=>?@H?E
>?;
1?.
conv2d_40_input?????????@@
p

 
? "&?#
?
0??????????
? ?
I__inference_sequential_10_layer_call_and_return_conditional_losses_140429w56789:;<=>?@??<
5?2
(?%
inputs?????????@@
p 

 
? "&?#
?
0??????????
? ?
I__inference_sequential_10_layer_call_and_return_conditional_losses_140504w56789:;<=>?@??<
5?2
(?%
inputs?????????@@
p

 
? "&?#
?
0??????????
? ?
.__inference_sequential_10_layer_call_fn_139224s56789:;<=>?@H?E
>?;
1?.
conv2d_40_input?????????@@
p 

 
? "????????????
.__inference_sequential_10_layer_call_fn_139487s56789:;<=>?@H?E
>?;
1?.
conv2d_40_input?????????@@
p

 
? "????????????
.__inference_sequential_10_layer_call_fn_140346j56789:;<=>?@??<
5?2
(?%
inputs?????????@@
p 

 
? "????????????
.__inference_sequential_10_layer_call_fn_140375j56789:;<=>?@??<
5?2
(?%
inputs?????????@@
p

 
? "????????????
$__inference_signature_wrapper_140317?56789:;<=>?@()w?t
? 
m?j
2
anchor(?%
anchor?????????@@
4
compare)?&
compare?????????@@"3?0
.
dense_32"?
dense_32?????????