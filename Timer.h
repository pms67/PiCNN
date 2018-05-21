#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <thread>

class Timer {
	
	//Microseconds
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;	
	
	//Seconds
	std::chrono::high_resolution_clock::time_point begin_s;
	std::chrono::high_resolution_clock::time_point end_s;	
	
	public:
	
		Timer() { }
		
		double elapsed(char type='s') {
			
			switch (type) {
				
				case 'u': //Microseconds
					return std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
					break;
				
				case 'm': //Milliseconds
					return std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
					break;
					
				default: //Seconds
					return std::chrono::duration_cast<std::chrono::seconds>(end-begin).count();
					break;
				
			}
			
		}
		
		void start() {
			
			begin = std::chrono::steady_clock::now();
			return;
						
		}
		
		double stop(char type='s') {
			
			end = std::chrono::steady_clock::now();
			return elapsed(type);				
			
		}
		
		void sleep(int s) {
			
			std::this_thread::sleep_for(s);
			return;
			
		}
	
};

#endif
