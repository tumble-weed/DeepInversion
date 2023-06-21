from argparse import Namespace
if False:
    score_and_edge_rotate = Namespace(**dict(
    edge_factor = 0.00001 *100,
    r_factor = 0 * 0.0001,
    loss_factor = 1  
    ))
if False:
    score_and_edge = Namespace(**dict(
    edge_factor = 0.001 *100,
    r_factor = 0 * 0.0001,
    loss_factor = 1  
    ))
if False:
    score_edge_and_normal = Namespace(**dict(
    edge_factor = 0.001 *100,
    normal_factor = 0.001 *100,
    r_factor = 0 * 0.0001,
    loss_factor = 1  
    ))
if True:
    # with score and r loss, edge
    score_r_edge_and_normal = Namespace(**dict(
    # edge_factor = 0.01*100*10,
    edge_factor = 100*10,
    r_factor = 100 * 0.0001,
    loss_factor = 1,
    normal_factor = 0.001 *100,
    ))
if False:
    # with score and r loss, edge
    score_r_and_edge = Namespace(**dict(
    edge_factor = 0.1 *100*10,
    r_factor = 100 * 0.0001,
    loss_factor = 1,
    normal_factor = 0,
    ))

# with score and r loss
score_and_r = Namespace(**dict(
edge_factor = 0*0.1 ,
r_factor = 100*0.0001,
loss_factor = 1
    
))

# with score loss
just_score = Namespace(**dict(
edge_factor = 0*0.1 ,
r_factor = 0*100 * 0.0001,
loss_factor = 1  
))
# with r_loss, levels 2, kl loss
edge_factor = 0.1 
r_factor = 100 * 0.0001
loss_factor = 1

# edge_factor = 10
# edge_factor = 100
# edge_factor = 10
# edge_factor = 1000 # mesh,softmax,level 4