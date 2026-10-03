use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// We use a global static Mutex so we can profile across the entire engine effortlessly.
lazy_static::lazy_static! {
    pub static ref PROFILER: Mutex<Profiler> = Mutex::new(Profiler::new());
}

pub struct Profiler {
    pub metrics: HashMap<String, Duration>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    pub fn record(&mut self, name: &str, duration: Duration) {
        let entry = self.metrics.entry(name.to_string()).or_insert(Duration::new(0, 0));
        *entry += duration;
    }

    pub fn print_summary(&self) {
        println!("\n📊 Operator Profiling Summary:");
        println!("{:<20} | {}", "Operator", "Time (ms)");
        println!("{:-<20}-+-{:-<15}", "", "");
        
        let mut total_ms = 0.0;
        let mut sorted_metrics: Vec<_> = self.metrics.iter().collect();
        // Sort by longest duration first
        sorted_metrics.sort_by(|a, b| b.1.cmp(a.1));

        for (name, duration) in sorted_metrics {
            let ms = duration.as_secs_f64() * 1000.0;
            total_ms += ms;
            println!("{:<20} | {:.2} ms", name, ms);
        }
        println!("{:-<20}-+-{:-<15}", "", "");
        println!("{:<20} | {:.2} ms", "TOTAL", total_ms);
    }
    
    pub fn reset(&mut self) {
        self.metrics.clear();
    }
}

/// A helper struct that measures the time between its creation and its destruction (Drop).
pub struct ProfileScope {
    name: String,
    start: Instant,
}

impl ProfileScope {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if let Ok(mut profiler) = PROFILER.lock() {
            profiler.record(&self.name, elapsed);
        }
    }
}