#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

#define HW_DIR "hw"
#define PROC_NAME "hw"
#define SCHEDULER_NAME "scheduler"
#define MEMORY_NAME "memory"

static struct proc_dir_entry *hw_dir;
static struct proc_dir_entry *scheduler_dir;
static struct proc_dir_entry *memory_dir;

// static int scheduler_show(struct seq_file *m, void *v);
// static int memory_show(struct seq_file *m, void *v);
// static int pid_info_open(struct inode *inode, struct file *file);
// static int pid_info_release(struct inode *inode, struct file *file);

MODULE_AUTHOR("Heewon Lim");
MODULE_DESCRIPTION("System Programming 2024 - 2019147503");

static int __init init(void) {
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
    memory_dir = proc_mkdir(MEMORY_DIR, hw_dir);
    if (!memory_dir) {
        pr_err("Failed to create /proc/%s/%s directory\n", HW_DIR, MEMORY_NAME);
        return -ENOMEM;
    }

    // PID 관련 파일 생성 (scheduler와 memory에 대해 각각)
    if (!proc_create_data(SCHEDULER_NAME, 0, scheduler_dir, &scheduler_fops, "scheduler")) {
        pr_err("Failed to create /proc/%s/%s/1234 file\n", HW_DIR, SCHEDULER_NAME);
        return -ENOMEM;
    }

    if (!proc_create_data(MEMORY_NAME, 0, memory_dir, &memory_fops, "memory")) {
        pr_err("Failed to create /proc/%s/%s/1234 file\n", HW_DIR, MEMORY_NAME);
        return -ENOMEM;
    }

    return 0; 
}

static void __exit exit(void) {
    // /proc 디렉토리와 파일 삭제
    remove_proc_entry(SCHEDULER_NAME, scheduler_dir);
    remove_proc_entry(MEMORY_NAME, memory_dir);
    remove_proc_entry(SCHEDULER_NAME, hw_dir);
    remove_proc_entry(MEMORY_NAME, hw_dir);
    remove_proc_entry(HW_DIR, NULL);

    pr_info("/proc/%s/%s and /proc/%s/%s and /proc/%s removed\n", HW_DIR, SCHEDULER_NAME, HW_DIR, MEMORY_NAME, HW_DIR);
}

module_init(init);
module_exit(exit);