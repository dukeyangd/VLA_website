import shlex
import unittest

import app


class SshCommandTests(unittest.TestCase):
    def test_ssh_disables_configured_forwardings(self):
        args = app.ssh_args("cluster_0", ["printf", "connected"])

        self.assertIn("ClearAllForwardings=yes", args)
        self.assertIn("BatchMode=yes", args)
        if app.SSH_CONFIG.is_file():
            self.assertEqual(args[1:3], ["-F", str(app.SSH_CONFIG)])

    def test_rsync_ssh_disables_configured_forwardings(self):
        rsync_options = app.rsync_ssh()

        self.assertEqual(rsync_options[0], "-e")
        ssh_command = shlex.split(rsync_options[1])
        self.assertIn("ClearAllForwardings=yes", ssh_command)
        self.assertIn("BatchMode=yes", ssh_command)
        if app.SSH_CONFIG.is_file():
            self.assertIn(str(app.SSH_CONFIG), ssh_command)


if __name__ == "__main__":
    unittest.main()
