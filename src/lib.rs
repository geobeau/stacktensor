mod tensor_batch;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor_batch::AppendError;

pub struct BatchRingBuffer {
    buffer: Vec<tensor_batch::TensorBatch>,
    tail: AtomicUsize,
    head: AtomicUsize,
    // The mask is used for efficient modulo arithmetic to wrap around the ring buffer.
    // The Problem:
    // When you have a ring buffer with n buffers, you need to convert a continuously incrementing index (0, 1, 2, 3, 4, 5...) into a buffer position (0, 1, 2, 3, 0, 1, 2, 3...).
    // Normally you'd use: buffer_idx = head % buffer_count
    // The Optimization:
    // If buffer_count is a power of 2 (e.g., 4, 8, 16), you can replace the slow modulo operation with a fast bitwise AND:
    // If buffer_count = 4 (which is 2^2)
    // mask = 4 - 1 = 3 = 0b0011

    // head = 0  → 0 & 0b0011 = 0
    // head = 1  → 1 & 0b0011 = 1
    // head = 2  → 2 & 0b0011 = 2
    // head = 3  → 3 & 0b0011 = 3
    // head = 4  → 4 & 0b0011 = 0  // Wraps around!
    // head = 5  → 5 & 0b0011 = 1
    // head = 6  → 6 & 0b0011 = 2
    mask: usize,
}

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

impl BatchRingBuffer {
    pub fn new(batches_nr: usize, tensor_size: usize, capacity: usize) -> BatchRingBuffer {
        // TODO: check batches_nr is power of 2
        if !is_power_of_two(batches_nr) {
            panic!("Buffer is not power of 2: {batches_nr}")
        }
        let mut buffer = Vec::with_capacity(batches_nr);
        for _ in 0..batches_nr {
            buffer.push(tensor_batch::TensorBatch::new(tensor_size, capacity));
        }
        BatchRingBuffer {
            buffer,
            tail: AtomicUsize::new(0),
            head: AtomicUsize::new(0),
            mask: batches_nr - 1,
        }
    }

    /// Append data to the current buffer, moving to next if full
    pub fn append(&self, data: &[u8]) -> Result<usize, AppendError> {
        loop {
            // head is where the current buffer is, as it is the hottest path.
            // taking an optimistic approach is preferable: try to append the batch
            // to this one and handle the failure with more atomics as needed.
            let head = self.head.load(Ordering::Acquire);

            // Wraps around like modulo, see the definition of self.mask for how it works
            let buffer_idx = head & self.mask;
            let buffer = &self.buffer[buffer_idx];

            // Try to append to current buffer
            match buffer.append(data) {
                Some(()) => return Ok(buffer_idx),
                None => {
                    // Buffer is full, try to move to next
                    let tail = self.tail.load(Ordering::Acquire);

                    // Check if we have space (at least one buffer available)
                    let new_head = (head + 1) & self.mask;
                    println!("mask {} {}", self.mask, (head + 1));
                    println!("buffer full, moving up {head} to {new_head})(tail: {tail})");
                    if tail == new_head {
                        return Err(AppendError::AllBuffersFull);
                    }

                    // Use CAS to ensure only one thread advances head
                    match self.head.compare_exchange_weak(
                        head,
                        new_head,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            // Successfully moved to next buffer, retry append
                            continue;
                        }
                        Err(_) => {
                            println!("Collision");
                            // Another thread moved head, retry with new head
                            continue;
                        }
                    }
                }
            }
        }
    }

    pub fn consume_buffer(&self) -> Result<(), AppendError> {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        if tail == head {
            return Err(AppendError::NoBufferReady);
        }
        self.buffer[tail].reset();

        let new_tail = (tail + 1) & self.mask;
        match self
            .tail
            .compare_exchange_weak(tail, new_tail, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                // Successfully moved to next buffer, retry append
                
                return Ok(());
            }
            Err(_) => {
                println!("Collision");
                // Another thread moved head, retry with new head
                return Err(AppendError::NoBufferReady);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let tensor_size = 1;
        let capacity = 1;
        let batches_nr = 4;

        let ringbatch = BatchRingBuffer::new(batches_nr, tensor_size, capacity);
        for i in 0..(capacity * batches_nr * 4) {
            println!("Inserting {i}");
            ringbatch.append(&vec![0u8; tensor_size]).unwrap();
            let consume = ringbatch.consume_buffer();
            println!("Consume {consume:?}")
        }
        // let final_data = stacked_tensors.get_data().unwrap();
        // assert_eq!(final_data, &data)
    }
}
