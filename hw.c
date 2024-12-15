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
    char comm[TASK_COMM_LEN];
    unsigned long pgd_base_addr;

    // 메모리 영역 정보
    struct {
        unsigned long start_vaddr;
        unsigned long end_vaddr;
        unsigned long start_paddr;
        unsigned long end_paddr;

        // Code 영역 전용 페이지 테이블 정보
        struct {
            unsigned long pgd_start;
            unsigned long pud_start;
            unsigned long pmd_start;
            unsigned long pte_start;
            unsigned long pgd_end;
            unsigned long pud_end;
            unsigned long pmd_end;
            unsigned long pte_end;
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
        if (strcmp(info->sched_type, "CFS") == 0) {
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

static struct memory_info* find_memory_info_by_pid(pid_t pid) {
    struct memory_info *info;

    list_for_each_entry(info, &memory_info_list, list) {
        if (info->pid == pid) {
            return info;
        }
    }

    return NULL;
}

// 메모리 정보 파일 읽기 핸들러
static int memory_show(struct seq_file *m, void *v) {
    spin_lock_irq(&my_lock);
    pid_t pid = *(pid_t *)m->private;
    struct memory_info *info = find_memory_info_by_pid(pid);

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
        seq_printf(m, "PGD base address: 0x%lx\n", info->pgd_base_addr);
        
        seq_printf(m, "Code Area\n");
        seq_printf(m, "- start (virtual): 0x%lx\n", info->areas[0].start_vaddr);
        seq_printf(m, "- start (PGD): 0x%lx, 0x%lx\n", info->areas[0].page_info.pgd_start, pgd_val(*(pgd_t *)info->areas[0].page_info.pgd_start));
        seq_printf(m, "- start (PUD): 0x%lx, 0x%lx\n", info->areas[0].page_info.pud_start, pud_val(*(pud_t *)info->areas[0].page_info.pud_start));
        seq_printf(m, "- start (PMD): 0x%lx, 0x%lx\n", info->areas[0].page_info.pmd_start, pmd_val(*(pmd_t *)info->areas[0].page_info.pmd_start));
        seq_printf(m, "- start (PTE): 0x%lx, 0x%lx\n", info->areas[0].page_info.pte_start, pte_val(*(pte_t *)info->areas[0].page_info.pte_start));
        seq_printf(m, "- start (physical): 0x%lx\n", info->areas[0].start_paddr);
        seq_printf(m, "- end (virtual): 0x%lx\n", info->areas[0].end_vaddr);
        seq_printf(m, "- end (physical): 0x%lx\n", info->areas[0].end_paddr);

        seq_printf(m, "Data Area\n");
        seq_printf(m, "- start (virtual): 0x%lx\n", info->areas[1].start_vaddr);
        seq_printf(m, "- start (physical): 0x%lx\n", info->areas[1].start_paddr);
        seq_printf(m, "- end (virtual): 0x%lx\n", info->areas[1].end_vaddr);
        seq_printf(m, "- end (physical): 0x%lx\n", info->areas[1].end_paddr);

        seq_printf(m, "Heap Area\n");
        seq_printf(m, "- start (virtual): 0x%lx\n", info->areas[2].start_vaddr);
        seq_printf(m, "- start (physical): 0x%lx\n", info->areas[2].start_paddr);
        seq_printf(m, "- end (virtual): 0x%lx\n", info->areas[2].end_vaddr);
        seq_printf(m, "- end (physical): 0x%lx\n", info->areas[2].end_paddr);

        seq_printf(m, "Stack Area\n");
        seq_printf(m, "- start (virtual): 0x%lx\n", info->areas[3].start_vaddr);
        seq_printf(m, "- start (physical): 0x%lx\n", info->areas[3].start_paddr);
        seq_printf(m, "- end (virtual): 0x%lx\n", info->areas[3].end_vaddr);
        seq_printf(m, "- end (physical): 0x%lx\n", info->areas[3].end_paddr);
    } else {
        seq_printf(m, "Invalid PID\n");
    }

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

// 스케줄러 정보 수집
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
    sched_info->start_time_ms = task->start_time / 1000000; // ns -> ms
    sched_info->utime_ms = task->utime / 1000000; // ns -> ms
    sched_info->stime_ms = task->stime / 1000000; // ns -> ms
    // sched_info->last_cpu = task_cpu(task);
    sched_info->last_cpu = task->recent_used_cpu;

    if (task->policy == SCHED_FIFO) {
        strcpy(sched_info->sched_type, "RT");
    } else if (task->policy == SCHED_NORMAL) {
        strcpy(sched_info->sched_type, "CFS");

         // CFS 전용 정보 수집
        struct sched_entity *se = &task->se;

        sched_info->weight = se->load.weight;
        sched_info->vruntime = se->vruntime;
    } else if (task->policy == SCHED_DEADLINE) {
        strcpy(sched_info->sched_type, "DL");
    } else if (task->policy == SCHED_IDLE) {
        strcpy(sched_info->sched_type, "IDLE");
    } else {
        strcpy(sched_info->sched_type, "UNKNOWN");
    }

    list_add_tail(&sched_info->list, &scheduler_info_list);
}

//please feaet virt_to_phys func
static unsigned long virt_to_phys(struct mm_struct *mm, unsigned long virt) {
    pgd_t *pgd = pgd_offset(mm, virt);
    p4d_t *p4d = p4d_offset(pgd, virt);
    pud_t *pud = pud_offset(p4d, virt);
    pmd_t *pmd = pmd_offset(pud, virt);
    pte_t *pte = pte_offset_kernel(pmd, virt);

    unsigned long page_address = pte_val(*pte) & PAGE_MASK;
    unsigned long page_offset = virt & ~PAGE_MASK;

    return page_address | page_offset;
}

// 메모리 관련 정보 수집
static void collect_memory_info(struct task_struct *task) {
    struct memory_info *info = kmalloc(sizeof(struct memory_info), GFP_ATOMIC);
    if (!info) {
        pr_err("Failed to allocate memory for memory_info\n");
        spin_unlock_irq(&my_lock);
        return;
    }

    struct mm_struct *mm = task->mm;
    if (mm) {
        info->pid = task->pid;
        get_task_comm(info->comm, task);
        
        info->pgd_base_addr = (unsigned long)mm->pgd;

        info->areas[0].start_vaddr = mm->start_code;
        info->areas[0].end_vaddr = mm->end_code;
        info->areas[0].start_paddr = virt_to_phys(mm, mm->start_code);
        info->areas[0].end_paddr = virt_to_phys(mm, mm->end_code);

        info->areas[0].page_info.pgd_start = pgd_offset(mm, mm->start_code);
        info->areas[0].page_info.pud_start = pud_offset(info->areas[0].page_info.pgd_start, mm->start_code);
        info->areas[0].page_info.pmd_start = pmd_offset(info->areas[0].page_info.pud_start, mm->start_code);
        info->areas[0].page_info.pte_start = pte_offset_kernel(info->areas[0].page_info.pmd_start, mm->start_code);

        info->areas[0].page_info.pgd_end = pgd_offset(mm, mm->end_code);
        info->areas[0].page_info.pud_end = pud_offset(info->areas[0].page_info.pgd_end, mm->end_code);
        info->areas[0].page_info.pmd_end = pmd_offset(info->areas[0].page_info.pud_end, mm->end_code);
        info->areas[0].page_info.pte_end = pte_offset_kernel(info->areas[0].page_info.pmd_end, mm->end_code);

        info->areas[1].start_vaddr = mm->start_data;
        info->areas[1].end_vaddr = mm->end_data;
        info->areas[1].start_paddr = virt_to_phys(mm, mm->start_data); 
        info->areas[1].end_paddr = virt_to_phys(mm, mm->end_data);

        info->areas[2].start_vaddr = mm->start_brk;
        info->areas[2].end_vaddr = mm->brk;
        info->areas[2].start_paddr = virt_to_phys(mm, mm->start_brk);
        info->areas[2].end_paddr = virt_to_phys(mm, mm->brk);

        info->areas[3].start_vaddr = mm->start_stack;
        info->areas[3].end_vaddr = find_vma(mm, mm->start_stack)->vm_end;
        // info->areas[3].end_vaddr = mm->start_stack - THREAD_SIZE;
        info->areas[3].start_paddr = virt_to_phys(mm, mm->start_stack);
        info->areas[3].end_paddr = virt_to_phys(mm, find_vma(mm, mm->start_stack)->vm_end);

        list_add_tail(&info->list, &memory_info_list);
    } else {
        pr_err("Failed to get mm_struct for PID %d\n", task->pid);
        kfree(info);
        spin_unlock_irq(&my_lock);
        return;
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
        collect_memory_info(task);
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