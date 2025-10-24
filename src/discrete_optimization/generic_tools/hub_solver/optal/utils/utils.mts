import * as zlib from 'zlib';
import * as fs from 'fs';

/* Small utilities used by most of the benchmarks */

/** Read a file, if necessary gunzip it, and convert into a string. */
export function readFile(filename: string): string {
  if (filename.endsWith(".gz"))
    return zlib.gunzipSync(fs.readFileSync(filename), {}).toString();
  else
    return fs.readFileSync(filename, "utf8");
}

/** Read a file, if necessary gunzip it, and convert into an array of numbers. */
export function readFileAsNumberArray(filename: string): number[] {
  return readFile(filename).trim().split(/\s+/).map(Number);
}

/**
 * Create model name from benchmark name and input data file name.
 * Basically connects those two strings and:
 * - Gets rid of common "data/" prefix in the filename.
 * - Replaces all slashes with underscores (so we can generate e.g. json export file names from model names).
 * - Gets rid of ".gz" suffix in the filename if present.
 * - Removes any other short suffix (2 or 3 characters) such as ".txt".
 */
export function makeModelName(benchmarkName: string, filename: string): string {
  return benchmarkName + '_' + filename.replaceAll(/[/\\]/g, '_').
    replace(/^data_/, '').
    replace(/\.gz$/, '').
    replace(/\.json$/, '').
    replace(/\....?$/, '');
}

/**
 * Get a boolean option from the command line arguments.
 *
 * @param option The option to look for, e.g. "--foo".
 * @param restArgs Input/output argument. The remaining command line arguments, without "--foo" if present.
 *
 * The result is true if the option is present, false otherwise.
 * The option does not take any value. I.e., we expect "--foo"
 * in the arguments, not "--foo 1".
 */
export function getBoolOption(option: string, restArgs: string[]): boolean {
  let index = restArgs.indexOf(option);
  if (index != -1) {
    restArgs.splice(index, 1);
    return true;
  }
  return false;
}

/**
 * Get a string option from the command line arguments.
 *
 * @param option The option to look for, e.g. "--foo".
 * @param defaultValue The default value if the option is not present.
 * @param restArgs Input/output argument. The remaining command line arguments, without "--foo xy" if present.
 *
 * The result is the value following the option in the arguments.
 * If the option is not present, the default value is returned.
 * The option takes a value. I.e., we expect "--foo xy".
 */
export function getStringOption(option: string, defaultValue: string, restArgs: string[]): string {
  let index = restArgs.indexOf(option);
  if (index == -1)
    return defaultValue;
  if (index + 1 == restArgs.length) {
    console.error(`Missing value for option ${option}`);
    process.exit(1);
  }
  let value = restArgs[index + 1];
  restArgs.splice(index, 2);
  return value;
}

/**
 * Get an integer option from the command line arguments.
 *
 * @param option The option to look for, e.g. "--foo".
 * @param defaultValue The default value if the option is not present.
 * @param restArgs Input/output argument. The remaining command line arguments, without "--foo nn" if present.
 *
 * The result is the value following the option in the arguments.
 * If the option is not present, the default value is returned.
 * The option takes a value. I.e., we expect "--foo nn".
 */
export function getIntOption(option: string, defaultValue: number, restArgs: string[]): number {
  let index = restArgs.indexOf(option);
  if (index == -1)
    return defaultValue;
  if (index + 1 == restArgs.length) {
    console.error(`Missing value for option ${option}`);
    process.exit(1);
  }
  let value = parseInt(restArgs[index + 1]);
  restArgs.splice(index, 2);
  return value;
}

/**
 * Get a float option from the command line arguments.
 *
 * @param option The option to look for, e.g. "--foo".
 * @param defaultValue The default value if the option is not present.
 * @param restArgs Input/output argument. The remaining command line arguments, without "--foo nn" if present.
 *
 * The result is the value following the option in the arguments.
 * If the option is not present, the default value is returned.
 * The option takes a value. I.e., we expect "--foo nn".
 */
export function getFloatOption(option: string, defaultValue: number, restArgs: string[]): number {
  let index = restArgs.indexOf(option);
  if (index == -1)
    return defaultValue;
  if (index + 1 == restArgs.length) {
    console.error(`Missing value for option ${option}`);
    process.exit(1);
  }
  let value = parseFloat(restArgs[index + 1]);
  restArgs.splice(index, 2);
  return value;
}
