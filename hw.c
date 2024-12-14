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

MODULE_AUTHOR("Heewon Lim");
MODULE_DESCRIPTION("System Programming 2024 - 2019147503");
MODULE_LICENSE("GPL");

#define HW_DIR "hw"
#define PROC_NAME "hw"
#define SCHEDULER_NAME "scheduler"
#define MEMORY_NAME "memory"
#define INTERVAL (5 * HZ)

// 디렉토리와 파일 관련 변수
static struct proc_dir_entry *hw_dir;
static struct proc_dir_entry *scheduler_dir;
static struct proc_dir_entry *memory_dir;
static struct timer_list timer;
static unsigned long last_collection_jiffies;

#define STUDENT_ID "2019147503"
#define STUDENT_NAME "Lim, Heewon"

static struct seq_operations scheduler_seq_ops = {
    .start = NULL,
    .next = NULL,
    .stop = NULL,
    .show = scheduler_show
};

static int scheduler_proc_open(struct inode *inode, struct file *file) {
    return seq_open(file, &scheduler_seq_ops);
}

static const struct proc_ops scheduler_proc_ops = {
    .proc_open = scheduler_proc_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = seq_release
};

static struct seq_operations memory_seq_ops = {
    .start = NULL,
    .next = NULL,
    .stop = NULL,
    .show = memory_show
};

static int memory_proc_open(struct inode *inode, struct file *file) {
    return seq_open(file, &memory_seq_ops);
}

static const struct proc_ops memory_proc_ops = {
    .proc_open = memory_proc_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = seq_release
};

static void collect_scheduler_info(struct seq_file *m, struct task_struct *task) {
    // 스케줄러 관련 정보 수집
    seq_printf(m, "Task: %s, PID: %d, PPID: %d, Priority: %d\n",
               task->comm, task->pid, task->real_parent->pid, task->prio);
    seq_printf(m, "Start time: %llu ms\n", task->start_time / 1000000);
    seq_printf(m, "User time: %llu ms, System time: %llu ms\n",
               task->utime / 1000000, task->stime / 1000000);
    // seq_printf(m, "Last CPU: %d\n", task->last_cpu);
    // 스케줄러 타입 정보 추가
    // if (task->sched_class == &fair_sched_class) {
    //     seq_printf(m, "Scheduler type: CFS\n");
    //     seq_printf(m, "Weight: %d, Virtual runtime: %llu\n",
    //                task->se.load.weight, task->se.vruntime);
    // } else if (task->sched_class == &rt_sched_class) {
    //     seq_printf(m, "Scheduler type: RT\n");
    // } else if (task->sched_class == &dl_sched_class) {
    //     seq_printf(m, "Scheduler type: DL\n");
    // } else if (task->sched_class == &idle_sched_class) {
    //     seq_printf(m, "Scheduler type: IDLE\n");
    // }
}

static void collect_memory_info(struct seq_file *m, struct task_struct *task) {
    // 메모리 관련 정보 수집
    struct mm_struct *mm = task->mm;
    if (mm) {
        seq_printf(m, "Code start: %lx, Code end: %lx\n", mm->start_code, mm->end_code);
        seq_printf(m, "Data start: %lx, Data end: %lx\n", mm->start_data, mm->end_data);
        seq_printf(m, "Heap start: %lx, Heap end: %lx\n", mm->start_brk, mm->brk);
        seq_printf(m, "Stack start: %lx\n", mm->start_stack);
    }
}

static void create_proc_files_for_tasks(void) {
    struct task_struct *task;
    char proc_name[16]; // 넉넉하게 잡음

    for_each_process(task) {
        if (task->flags & PF_KTHREAD)
            continue; // 커널 스레드는 제외

        snprintf(proc_name, sizeof(proc_name), "%d", task->pid);
        printk("Creating /proc/%s/%s and /proc/%s/%s\n", HW_DIR, SCHEDULER_NAME, HW_DIR, MEMORY_NAME);
        proc_create_data(proc_name, 0644, scheduler_dir, &scheduler_proc_ops, &task->pid);
        proc_create_data(proc_name, 0644, memory_dir, &memory_proc_ops, &task->pid);
    }
}

static void timer_callback(struct timer_list *t) {
    printk("Timer callback called\n");
    create_proc_files_for_tasks();

    // 마지막 수집 시점의 jiffies 저장
    last_collection_jiffies = jiffies;

    // 타이머 재설정
    mod_timer(&timer, jiffies + INTERVAL);
}

// 스케줄러 정보 파일 읽기 핸들러
static int scheduler_show(struct seq_file *m, void *v) {
    pid_t pid = *(pid_t *)v;
    struct task_struct *task;

    // 기본 정보 출력
    seq_printf(m, "[System Programming Assignment (2024)]\n");
    seq_printf(m, "ID: %s}\n", STUDENT_ID);
    seq_printf(m, "Name: %s\n", STUDENT_NAME);
    
    // Uptime 정보 출력
    seq_printf(m, "Current Uptime (s): %lu\n",
               (jiffies - INITIAL_JIFFIES) / HZ);
    
    // 태스크 찾기
    task = pid_task(find_vpid(pid), PIDTYPE_PID);
    if (!task) {
        seq_printf(m, "Invalid PID\n");
        return 0;
    }

    // 상세 스케줄러 정보 출력
    seq_printf(m, "--------------------------------------------------\n");
    seq_printf(m, "Command: %s\n", task->comm);
    seq_printf(m, "PID: %d\n", task->pid);
    seq_printf(m, "--------------------------------------------------\n");
    seq_printf(m, "PPID: %d\n", task->real_parent->pid);
    seq_printf(m, "Priority: %d\n", task->prio);
    
    // 추가 정보 출력 로직 필요...

    return 0;
}


// 메모리 정보 파일 읽기 핸들러
static int memory_show(struct seq_file *m, void *v) {
    pid_t pid = *(pid_t *)v;
    struct task_struct *task;
    struct mm_struct *mm;

    // 기본 정보 및 Uptime 출력 (스케줄러와 동일)
    
    // 태스크 및 메모리 구조체 찾기
    task = pid_task(find_vpid(pid), PIDTYPE_PID);
    if (!task || !(mm = task->mm)) {
        seq_printf(m, "Invalid PID or No Memory Map\n");
        return 0;
    }

    // 메모리 영역별 정보 출력
    // Code, Data, Heap, Stack 각 영역의 가상/물리 주소 변환 및 출력
    // PGD, PUD, PMD, PTE 정보 추출 및 출력
    
    return 0;
}

static int __init hw_init(void) {
    // hw 디렉토리 생성
    hw_dir = proc_mkdir(HW_DIR, NULL);
    if (!hw_dir) {
        pr_err("Failed to create /proc/%s directory\n", HW_DIR);
        return -ENOMEM;
    }

    // scheduler 디렉토리 생성
    scheduler_dir = proc_mkdir(SCHEDULER_NAME, hw_dir);
    if (!scheduler_dir) {
        pr_err("Failed to create /proc/%s/%s directory\n", HW_DIR, SCHEDULER_NAME);
        return -ENOMEM;
    }

    // memory 디렉토리 생성
    memory_dir = proc_mkdir(MEMORY_NAME, hw_dir);
    if (!memory_dir) {
        pr_err("Failed to create /proc/%s/%s directory\n", HW_DIR, MEMORY_NAME);
        return -ENOMEM;
    }

    // 타이머 초기화
    timer_setup(&timer, timer_callback, 0);
    mod_timer(&timer, jiffies);

    pr_info("/proc/%s/%s and /proc/%s/%s created\n", HW_DIR, SCHEDULER_NAME, HW_DIR, MEMORY_NAME);
    return 0; 
}

static void __exit hw_exit(void) {
    del_timer_sync(&timer);
    remove_proc_subtree(HW_DIR, NULL);

    pr_info("/proc/%s/%s and /proc/%s/%s and /proc/%s removed\n", HW_DIR, SCHEDULER_NAME, HW_DIR, MEMORY_NAME, HW_DIR);
}

module_init(hw_init);
module_exit(hw_exit);