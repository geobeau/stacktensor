# Stacktensor

This repository is testing ground for lockless batching system for inference. It will eventually
be merged into https://github.com/geobeau/inference-server if I am not too lazy.

# Goal

For inference, especially on GPU, it is better to batch many inputs before start an inference
session. As the most expensive component of the target server type is most likely the GPU, it is important
to achieve the highest throughput and ensuring the rest of the stack will not be bottlenecks.


Models defines inputs, typically Tensors (multi-dimentionals arrays), that can be batch. In
practice, batching is mostly happending fixed-size array into a bigger one. The size is defined by the shape.
E.g. A tensor of shape [1, 32, 32] is 1024 numbers, can be part of a batch of [128, 32, 32].

The second underlying goal is to have a system that can handle a model with very low inference per second
and scale it to 1M or more inferences per second.

The most efficient (from a GPU utilization), is to have all inference requests send to a queue and have
a consummer form the batch and dispatch it to an executor as soon as possible. This will make the batch
forming thread the bottleneck. This bottleneck can be mitigated by creating multiple queues and sending
requests in round robin. But if you do that, when the throughput is low it will longer to fill batches
(as you are trying to fill N batches in parallel), increasing latency (or imcomplete batches are sent anyway
after a timeout and GPU utilization will suffer).

# Idea

Instead of using queues, I want to experiment with concurrent datastructure that coordinate writing
using atomics.
Each threads will try to access the current buffer, reserve a slot in the batch with atomic operations,
write at the appropriate slot then wait for a response.
Atomics have a performance costs, but I hope it is less than queues (that relies on atomics as well)
or locking.

The code is split in 2 data structures:
- the batch itself
- a ringbuffer of batches (so you can write to another batch if the current one is full)



