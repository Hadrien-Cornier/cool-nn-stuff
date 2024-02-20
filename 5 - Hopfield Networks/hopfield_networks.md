
# Hopfield networks are transformers

From the paper : Hopfield Networks are alll you need

Hofield networks are basically graphs whose values can settle into an energy function minima as they describe trajectories that lower their energy function.

Another basic way to see this is that it is a memory unit biologically inspired that can map an incomplete pattern into a complete one

This can be used for superresolution or reconstruction of obstructed images. 

What are the uses for text tasks of Hopfield networks ? 

So far I know that they can be used to reconstruct images that are partially masked (this is similar to masked language modeling ) or corrupted (this is similar to denoising)

Could we do the same for a recommendation/classification problem ? Could we obscure the labels and then ask the hopfield network to retrieve them ? Is there currecntly any situation where we are using transformers but instead could use a hopfield network ? 

It is kind of a routing layer that can take an input embedding and convert it into an embedding in another space, in a deterministic manner (almost surely).

But does it have strucural advantages compared to the transformer ? 

It has exponential convergence rate and converges really fast to the response but it still takes multiple steps to accomplish

This introduces latency compared to a transformer 

One attribute it has is that it looks way more like a human brain. Maybe when we look for something we create an incomplete pattern in our minds that flickers into equilibrium on a specific memory ?

If our brains did it, maybe it is good for AGI ?

#AssociativeMemory This just means that the end state is associated with the input (DUH), I don’t know what’s so special, it’s basically just a function



___
# Standard associative Memory, Classical Hopfield Network

Formal arguments exist for the link between Hopfield networks and Transformers.
For the case of denoising, it is less explored currently.


Hopfield created this concept in the 1982 
Amit et al. 1985

Network of N nodes which can take a binary value $-1$ or $+1$
$$E=-\Sigma_{i,j=1}^N \sigma_iT_{ij}\sigma_j$$
$$T_{i,j}=\Sigma_{\mu=1}^K \xi_i^{\mu} \xi_j^{\mu}$$

With : 
- $\sigma_i$ dynamical variables
- $\xi_i^{\mu}$ memorized patterns
- $N$ number of neurons
- $K$ number of memories

$E=-\Sigma_{\mu=1}^K(\Sigma_{i=1}^N \xi_i^{\mu} \sigma_i)^2$


The limiation of this technique is that if we try to memorize more and more patterns at the end we will encounter problems

We can always store more in the network but retrieval can fail because it can be interfered with other memories

This failure mode happens when 2 local minimas that are close to each other yield a local minima which is the middle of both of them : superposition of states and confusion happens.

In statistical physics this concept of bad local minimum is often associated with spin glass states

$K^{max} \approx 0.14N$

This sucks. Linear scaling is terrible. We want exponential scaling of memory capacity, not linear scaling.


  ___
# Dense Associative Memory, Modern Hopfield Network

Krotov & Hopfield 2016
Demircigil et al. 2017

$$E=-\Sigma_{\mu=1}^KF(\Sigma_{i=1}^N\xi_i^{\mu}\sigma_i)$$



Energy function that has higher term orders in it
In the old hopfield network we square the deviation from the target
This formulation is what causes the linear memory scaling


Now we will change that by picking a very fast rising energy function F like $x^n$ with $n>2$ or $e^{x}$


Picking a more steep memory function makes the energy landscape more variable and distinguishes the local minimas more by pulling up deviations faster


In this case 
$$K^{max}(F:x^n) \approx \alpha_n N^{n-1}$$
$$K^{max}(F:e^x) \approx e^{\alpha N} , \alpha < \ln(2)/2$$
So basically polynomial scaling of the loss => polynomial scaling of the memory capacity
Exponential scaling of the loss => exponential scaling of the memory capacity



So this is the main progress that happened recently on Hopfield networks.
