#[cfg(feature = "loom")]
use loom::cell::UnsafeCell;
#[cfg(feature = "loom")]
use loom::sync::atomic::{fence, AtomicUsize, Ordering};

use std::cell::RefCell;

#[cfg(not(feature = "loom"))]
use std::cell::UnsafeCell;
#[cfg(not(feature = "loom"))]
use std::sync::atomic::{fence, AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendError {
    AllBuffersFull,
}

enum BatchState {
    Writtable,
    Written,
    Consumed,
}

pub struct TensorBatch {
    #[cfg(feature = "loom")]
    data: Vec<UnsafeCell<Vec<u8>>>,
    #[cfg(not(feature = "loom"))]
    data: UnsafeCell<Vec<u8>>,
    capacity: usize,
    tensor_size: usize,
    reserved_slots: AtomicUsize,
    written_slots: AtomicUsize,
    state: RefCell<BatchState>,
}

unsafe impl Send for TensorBatch {}
unsafe impl Sync for TensorBatch {}

impl TensorBatch {
    pub fn new(tensor_size: usize, capacity: usize) -> TensorBatch {
        #[cfg(feature = "loom")]
        {
            let mut data = Vec::with_capacity(capacity);
            for _ in 0..capacity {
                data.push(UnsafeCell::new(vec![0; tensor_size]))
            }
            TensorBatch {
                data,
                tensor_size,
                capacity,
                reserved_slots: AtomicUsize::new(0),
                written_slots: AtomicUsize::new(0),
                state: RefCell::new(BatchState::Writtable),
            }
        }
        #[cfg(not(feature = "loom"))]
        {
            let data = UnsafeCell::new(vec![0; tensor_size * capacity]);
            TensorBatch {
                data,
                tensor_size,
                capacity,
                reserved_slots: AtomicUsize::new(0),
                written_slots: AtomicUsize::new(0),
                state: RefCell::new(BatchState::Writtable),
            }
        }
    }

    pub fn append(&self, data: &[u8]) -> Option<()> {
        assert_eq!(data.len(), self.tensor_size);
        let idx = self.reserved_slots.fetch_add(1, Ordering::SeqCst);
        if idx >= self.capacity {
            return None;
        }

        #[cfg(feature = "loom")]
        {
            self.data[idx].with_mut(|ptr| unsafe {
                let slice = &mut (&mut (*ptr))[0..self.tensor_size];
                slice.copy_from_slice(data);
            });
        }

        #[cfg(not(feature = "loom"))]
        unsafe {
            let start = idx * self.tensor_size;
            let end = (idx + 1) * self.tensor_size;

            // Isolation of portions of the vector is guaranteed by reserved_slots atomics
            let slice = &mut (&mut (*self.data.get()))[start..end];
            slice.copy_from_slice(data);
        }
        fence(Ordering::Release);

        // Mark this slot as written
        let slot_written = self.written_slots.fetch_add(1, Ordering::Release);
        if slot_written + 1 == self.capacity {
            self.state.replace(BatchState::Written);
        }
        Some(())
    }

    pub fn get_data(&self) -> Option<&Vec<u8>> {
        if self.written_slots.load(Ordering::Acquire) == self.capacity {
            #[cfg(feature = "loom")]
            {
                // For loom, we can't safely return a reference
                // This is a limitation - in real code you'd want proper synchronization
                None
            }

            #[cfg(not(feature = "loom"))]
            unsafe {
                Some(&*self.data.get())
            }
        } else {
            None
        }
    }

    /// Check if this buffer is full
    pub fn is_full(&self) -> bool {
        self.reserved_slots.load(Ordering::Acquire) >= self.capacity
    }

    /// Get the number of slots currently reserved
    pub fn reserved_count(&self) -> usize {
        self.reserved_slots.load(Ordering::Acquire)
    }

    /// Get the number of slots fully written
    pub fn written_count(&self) -> usize {
        self.written_slots.load(Ordering::Acquire)
    }

    /// Check if all reserved slots have been written
    pub fn is_ready(&self) -> bool {
        let written = self.written_slots.load(Ordering::Acquire);
        written == self.capacity
    }

    /// Reset the buffer for reuse
    pub fn reset(&self) {
        self.reserved_slots.store(0, Ordering::Release);
        self.written_slots.store(0, Ordering::Release);
    }

    /// Get the capacity of this buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the tensor size
    pub fn tensor_size(&self) -> usize {
        self.tensor_size
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "loom")]
    use loom::sync::Arc;
    #[cfg(feature = "loom")]
    use loom::thread;

    #[cfg(not(feature = "loom"))]
    #[test]
    fn it_works() {
        let tensor_size = 8;
        let capacity = 8;
        let data: Vec<u8> = (0..tensor_size * capacity).map(|i| i as u8).collect();

        let stacked_tensors = TensorBatch::new(tensor_size, capacity);
        for i in 0..capacity {
            stacked_tensors.append(&data[i * tensor_size..(i + 1) * tensor_size]);
        }
        let final_data = stacked_tensors.get_data().unwrap();
        assert_eq!(final_data, &data)
    }

    #[test]
    #[cfg(feature = "loom")]
    fn loom_concurrent_append_two_threads() {
        loom::model(|| {
            let tensor_size = 2;
            let capacity = 2;

            let stacked = Arc::new(TensorBatch::new(tensor_size, capacity));

            let stacked1 = stacked.clone();
            let stacked2 = stacked.clone();
            let stacked3 = stacked.clone();
            // let stacked4 = stacked.clone();

            let t1 = thread::spawn(move || {
                let data = vec![1u8; 2];
                stacked1.append(&data)
            });

            let t2 = thread::spawn(move || {
                let data = vec![2u8; 2];
                stacked2.append(&data)
            });

            let t3 = thread::spawn(move || {
                let data = vec![2u8; 2];
                stacked3.append(&data)
            });

            // let t4 = thread::spawn(move || {
            //     let data = vec![2u8; 2];
            //     stacked4.append(&data)
            // });

            let r1 = t1.join().unwrap();
            let r2 = t2.join().unwrap();
            let r3 = t3.join().unwrap();
            // let r4 = t4.join().unwrap();
            let mut some = 0;
            let mut none = 0;
            [r1, r2, r3].into_iter().for_each(|x| {
                match x {
                    Some(_) => some+=1,
                    None => none+=1,
                }
            });
            assert!(some == 2);
            assert!(none == 1);

            // Check that all slots are written
            assert_eq!(stacked.written_count(), capacity);
            assert!(stacked.is_ready());

        });
    }
}
