#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/timer.h>
#include <linux/jiffies.h>
#include <linux/uaccess.h>
#include <linux/mm.h>
#include <linux/mm_types.h>
#include <linux/pid.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/interrupt.h>
#include <linux/vmalloc.h>
#include <linux/mutex.h>

MODULE_AUTHOR("Heewon Lim");
MODULE_DESCRIPTION("System Programming 2024 - 2019147503");
MODULE_LICENSE("GPL");

#define HW_DIR "hw"
#define PROC_NAME "hw"
#define SCHEDULER_NAME "scheduler"
#define MEMORY_NAME "memory"
#define INTERVAL (5 * HZ)
DEFINE_SPINLOCK(my_lock);

// 디렉토리와 파일 관련 변수
static struct proc_dir_entry *hw_dir;
static struct proc_dir_entry *scheduler_dir;
static struct proc_dir_entry *memory_dir;
static struct timer_list timer;
static unsigned long last_collection_jiffies;

#define STUDENT_ID "2019147503"
#define STUDENT_NAME "Lim, Heewon"

struct scheduler_info {
    pid_t pid;
    char comm[TASK_COMM_LEN];
    pid_t ppid;
    int prio;
    unsigned long start_time_ms;
    unsigned long utime_ms;
    unsigned long stime_ms;
    int last_cpu;
    char sched_type[16];
    
    // CFS 전용 정보
    unsigned long weight;
    unsigned long long vruntime;
    
    struct list_head list;
};

// 태스크 메모리 정보 구조체
struct memory_info {
    pid_t pid;
    
    // 메모리 영역 정보
    struct {
        unsigned long start_vaddr;
        unsigned long end_vaddr;
        unsigned long start_paddr;
        unsigned long end_paddr;
        
        // Code 영역 전용 페이지 테이블 정보
        struct {
            unsigned long pgd_addr;
            unsigned long pgd_val;
            unsigned long pud_addr;
            unsigned long pud_val;
            unsigned long pmd_addr;
            unsigned long pmd_val;
            unsigned long pte_addr;
            unsigned long pte_val;
        } page_info;
    } areas[4];  // Code, Data, Heap, Stack 순서

    struct list_head list;
};

// 글로벌 리스트 헤드
LIST_HEAD(scheduler_info_list);
LIST_HEAD(memory_info_list);

static struct scheduler_info* find_scheduler_info_by_pid(pid_t pid) {\
    struct scheduler_info *info;

    list_for_each_entry(info, &scheduler_info_list, list) {
        if (info->pid == pid) {
            return info;
        }
    }

    return NULL; // 해당 pid를 가진 구조체를 찾지 못한 경우
}

// 스케줄러 정보 파일 읽기 핸들러
static int scheduler_show(struct seq_file *m, void *v) {
    spin_lock_irq(&my_lock);
    pid_t pid = *(pid_t *)m->private;
    struct scheduler_info *info = find_scheduler_info_by_pid(pid);

    if (info) {
        seq_printf(m, "[System Programming Assignment (2024)]\n");
        seq_printf(m, "ID: %s\n", STUDENT_ID);
        seq_printf(m, "Name: %s\n", STUDENT_NAME);
        seq_printf(m, "Current Uptime (s): %lu\n", (jiffies - INITIAL_JIFFIES) / HZ);
        seq_printf(m, "Last Collection Uptime (s): %lu\n", (last_collection_jiffies - INITIAL_JIFFIES) / HZ);
        seq_printf(m, "--------------------------------------------------\n");
        seq_printf(m, "Command: %s\n", info->comm);
        seq_printf(m, "PID: %d\n", pid);
        seq_printf(m, "--------------------------------------------------\n");
        seq_printf(m, "PPID: %d\n", info->ppid);
        seq_printf(m, "Priority: %d\n", info->prio);
        seq_printf(m, "Start time (ms): %lu\n", info->start_time_ms);
        seq_printf(m, "User mode time (ms): %lu\n", info->utime_ms);
        seq_printf(m, "System mode time (ms): %lu\n", info->stime_ms);
        seq_printf(m, "Last CPU: %d\n", info->last_cpu);
        seq_printf(m, "Scheduler: %s\n", info->sched_type);
        if (info->sched_type == "CFS") {
            seq_printf(m, "Weight: %lu\n", info->weight);
            seq_printf(m, "Virtual Runtime: %llu\n", info->vruntime);
        }
    } else {
        seq_printf(m, "Invalid PID\n");
    }

    spin_unlock_irq(&my_lock);

    return 0;
}

static int scheduler_proc_open(struct inode *inode, struct file *file) {
    return single_open(file, scheduler_show, pde_data(inode));
}

static const struct proc_ops scheduler_fops = {
    .proc_open = scheduler_proc_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = seq_release
};

// 메모리 정보 파일 읽기 핸들러
static int memory_show(struct seq_file *m, void *v) {
    spin_lock_irq(&my_lock);
    // struct memory_info *mem_info;
    // pid_t pid = *(pid_t *)v;
    // struct task_struct *task;
    // struct mm_struct *mm;    

    seq_printf(m, "[System Programming Assignment (2024)]\n");
    seq_printf(m, "ID: %s\n", STUDENT_ID);
    seq_printf(m, "Name: %s\n", STUDENT_NAME);
    seq_printf(m, "Current Uptime (s): %lu\n", (jiffies - INITIAL_JIFFIES) / HZ);
    seq_printf(m, "Last Collection Uptime (s): %lu\n", (last_collection_jiffies - INITIAL_JIFFIES) / HZ);
    seq_printf(m, "--------------------------------------------------\n");

    // 태스크 및 메모리 구조체 찾기
    // task = pid_task(find_vpid(pid), PIDTYPE_PID);
    // if (!task || !(mm = task->mm)) {
    //     seq_printf(m, "Invalid PID or No Memory Map\n");
    //     return 0;
    // }

    // 메모리 영역별 정보 출력
    // Code, Data, Heap, Stack 각 영역의 가상/물리 주소 변환 및 출력
    // PGD, PUD, PMD, PTE 정보 추출 및 출력

    spin_unlock_irq(&my_lock);

    return 0;
}

static int memory_proc_open(struct inode *inode, struct file *file) {
    return single_open(file, memory_show, pde_data(inode));
}

static const struct proc_ops memory_fops = {
    .proc_open = memory_proc_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = seq_release
};

static void collect_scheduler_info(struct task_struct *task) {    
    struct scheduler_info *sched_info = kmalloc(sizeof(struct scheduler_info), GFP_ATOMIC);
    if (!sched_info) {
        pr_err("Failed to allocate memory for scheduler_info\n");
        spin_unlock_irq(&my_lock);
        return;
    }

    sched_info->pid = task->pid;
    get_task_comm(sched_info->comm, task);
    sched_info->ppid = task->parent->pid;
    sched_info->prio = task->prio;
    sched_info->start_time_ms = task->start_time / HZ;
    sched_info->utime_ms = task->utime / HZ;
    sched_info->stime_ms = task->stime / HZ;
    // sched_info->last_cpu = task_cpu(task);
    sched_info->last_cpu = task->recent_used_cpu;

    if (task->sched_class == SCHED_FIFO) {
        strcpy(sched_info->sched_type, "RT");
    } else if (task->sched_class == SCHED_NORMAL) {
        strcpy(sched_info->sched_type, "CFS");
    } else if (task->sched_class == SCHED_DEADLINE) {
        strcpy(sched_info->sched_type, "DL");
    } else if (task->sched_class == SCHED_IDLE) {
        strcpy(sched_info->sched_type, "IDLE");
    } else {
        strcpy(sched_info->sched_type, "UNKNOWN");
    }

    // CFS 전용 정보 수집
    if (task->sched_class == SCHED_NORMAL) {
        struct sched_entity *se = &task->se;
        sched_info->weight = se->load.weight;
        sched_info->vruntime = se->vruntime;
    }

    list_add_tail(&sched_info->list, &scheduler_info_list);
}

static void collect_memory_info(struct seq_file *m, struct task_struct *task) {
    // 메모리 관련 정보 수집
    struct mm_struct *mm = task->mm;
    if (mm) {
        // seq_printf(m, "Code start: %lx, Code end: %lx\n", mm->start_code, mm->end_code);
        // seq_printf(m, "Data start: %lx, Data end: %lx\n", mm->start_data, mm->end_data);
        // seq_printf(m, "Heap start: %lx, Heap end: %lx\n", mm->start_brk, mm->brk);
        // seq_printf(m, "Stack start: %lx\n", mm->start_stack);
    }
}

void remove_files_and_clear_info_list(void) {
    struct scheduler_info *scehd_entry, *scehd_tmp;
    struct memory_info *mem_entry, *mem_tmp;

    list_for_each_entry_safe(scehd_entry, scehd_tmp, &scheduler_info_list, list) {
        char proc_name[16];
        snprintf(proc_name, sizeof(proc_name), "%d", scehd_entry->pid);
        remove_proc_entry(proc_name, scheduler_dir);
        list_del(&scehd_entry->list);
        kfree(scehd_entry);
    }

    list_for_each_entry_safe(mem_entry, mem_tmp, &memory_info_list, list) {
        char proc_name[16];
        snprintf(proc_name, sizeof(proc_name), "%d", mem_entry->pid);
        remove_proc_entry(proc_name, memory_dir);  // memory_dir은 해당 디렉토리
        list_del(&mem_entry->list);
        kfree(mem_entry);
    }
}

void timer_callback(struct timer_list* timer) {
    struct task_struct* task;
    struct memory_info *mem_info, *mem_tmp;
    char proc_name[16]; // PID 최대 길이

    spin_lock_irq(&my_lock);

    remove_files_and_clear_info_list();

    rcu_read_lock();
    for_each_process(task) {
        if (task->flags & PF_KTHREAD)
            continue; // 커널 스레드는 제외

        snprintf(proc_name, sizeof(proc_name), "%d", task->pid);
        proc_create_data(proc_name, 0644, scheduler_dir, &scheduler_fops, &task->pid);
        proc_create_data(proc_name, 0644, memory_dir, &memory_fops, &task->pid);

        collect_scheduler_info(task);
    }
    rcu_read_unlock();

    // 마지막 수집 시점의 jiffies 저장
    last_collection_jiffies = jiffies;

    // 타이머 재설정
    mod_timer(timer, jiffies + INTERVAL);

    spin_unlock_irq(&my_lock);
}

static int __init hw_init(void) {
    // hw 디렉토리 생성
    hw_dir = proc_mkdir(HW_DIR, NULL);
    if (!hw_dir) {
        pr_err("Failed to create /proc/%s directory\n", HW_DIR);
        spin_unlock_irq(&my_lock);
        return -ENOMEM;
    }

    // scheduler 디렉토리 생성
    scheduler_dir = proc_mkdir(SCHEDULER_NAME, hw_dir);
    if (!scheduler_dir) {
        pr_err("Failed to create /proc/%s/%s directory\n", HW_DIR, SCHEDULER_NAME);
        spin_unlock_irq(&my_lock);
        return -ENOMEM;
    }

    // memory 디렉토리 생성
    memory_dir = proc_mkdir(MEMORY_NAME, hw_dir);
    if (!memory_dir) {
        pr_err("Failed to create /proc/%s/%s directory\n", HW_DIR, MEMORY_NAME);
        spin_unlock_irq(&my_lock);
        return -ENOMEM;
    }

    // 타이머 초기화
    timer_setup(&timer, timer_callback, 0);
    mod_timer(&timer, jiffies);

    pr_info("module inserted\n");

    spin_unlock_irq(&my_lock);
    return 0;
}

static void __exit hw_exit(void) {
    spin_lock_irq(&my_lock);

    del_timer_sync(&timer);
    remove_proc_subtree(HW_DIR, NULL);

    spin_unlock_irq(&my_lock);
}

module_init(hw_init);
module_exit(hw_exit);